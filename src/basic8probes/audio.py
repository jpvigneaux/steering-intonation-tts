"""Encoder wrappers that turn an audio file into a frame sequence.

One :class:`AudioManager` subclass per encoder. Each exposes the same two entry points:

``get_latent_for_last_word``
    the frames spanning the final (nuclear) word, located from a forced-alignment TextGrid;
``get_latents_for_utterance``
    every frame of the recording, which the whole-utterance sequence probes consume.

The ``embedding_type`` argument selects the representation:

``unquantized``
    the encoder output. For Mimi this is *not* the pre-quantization latent -- see the note on
    :meth:`MimiAudioManager._get_unquantized_latents_for_last_word`.
``prequantized`` (Mimi) / ``dequantized`` (SpeechTokenizer)
    the two sides of the quantizer, whose gap measures the cost of quantization.
``codebook``
    the codeword vectors of one residual level, selected with ``codebook_index``.

Frame rates differ by encoder and are the caller's responsibility: 12.5 Hz for Mimi, 50 Hz for
SpeechTokenizer, WavLM and HuBERT. Passing the wrong rate silently misaligns the word crop.
"""

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from praatio import textgrid
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel
from speechtokenizer import SpeechTokenizer

from . import config

WAVLM_MODEL_ID = "microsoft/wavlm-large"
HUBERT_MODEL_ID = "facebook/hubert-base-ls960"

# Frame rate of each encoder's latent sequence, in Hz.
LATENT_RATE_HZ = {"mimi": 12.5, "st": 50.0, "wavlm": 50.0, "hubert": 50.0}

# Classes to extract latent representation from audio + textgrid
class AudioManager:
    def __init__(self, device):
        pass

    def get_latent_for_last_word(self,audio_path: str, embedding_type: str, textgrid_path = None, codebook_index=0, latent_rate_hz=12.5):
        pass
    
class MimiAudioManager(AudioManager):
    def __init__(self, device, num_codebooks: int | None = None):
        """``num_codebooks`` selects how many residual levels the codec exposes.

        The default of 8 is what the probing experiments use and what ``get_mimi`` returns.
        A generative model built on Mimi may configure more -- the CSM checkpoint uses 32 --
        and decoding that model's own token stream requires matching it, so the count is a
        parameter rather than a constant.
        """
        self.device = device
        config.hf_login_if_configured()
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        if num_codebooks is not None:
            self.mimi.set_num_codebooks(num_codebooks)
        self.sample_rate = self.mimi.sample_rate
        
    def _get_token_indices_for_word(self, textgrid_path: Path, latent_rate_hz=12.5, word_position: int = -1):
        tier_name="words"
        # Get time stamps from TExtGrid
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals= False)
        words_tier = tg.getTier(tier_name)

        # Transform time stamps into mimi time step indices
        start, end, _ = words_tier.entries[word_position]
        t_start = int(np.floor(start * latent_rate_hz))
        t_end = int(np.ceil(end * latent_rate_hz))
        if word_position == -1: 
            return np.arange(t_start, t_end+1)
        else:
            return np.arange(t_start, t_end)

    def _get_unquantized_latents_for_last_word(self,audio_tensor, word_indices):
        # NOTE: `encode_to_latent` has signature (x, quantize: bool = True), so this returns
        # decode_latent(quantizer.encode(emb)) -- the latent reconstructed from Mimi's 8
        # residual levels, which is what the DECODER consumes, not the encoder's
        # pre-quantization output. Every "unquantized" Mimi embedding in this repository was
        # produced by this call, and it is kept exactly as-is so those results stay
        # reproducible; use embedding_type="prequantized" for the true pre-quantization latent.
        with torch.no_grad():
            uq_embeddings = self.mimi.encode_to_latent(audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device))
        truncated_word_indices = [ idx for idx in word_indices if idx < uq_embeddings.shape[2]]
        return uq_embeddings[0, :, truncated_word_indices], truncated_word_indices

    def _get_prequantized_latents_for_last_word(self, audio_tensor, word_indices):
        """The encoder's output BEFORE the quantizer: encode_to_latent(x, quantize=False).

        This is the object SpeechTokenizer's `model.encoder(x)` and WavLM's
        `last_hidden_state` correspond to, so it is what makes the three encoders comparable.
        It lives in a different space from the dequantized latent above (norm/frame ~1.5
        against ~67, and near-orthogonal coordinates, since the quantizer's input_proj and
        output_proj are independent learned convolutions).
        """
        with torch.no_grad():
            pq = self.mimi.encode_to_latent(
                audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device), quantize=False)
        truncated_word_indices = [idx for idx in word_indices if idx < pq.shape[2]]
        return pq[0, :, truncated_word_indices], truncated_word_indices


    def _get_codebook_latents_for_last_word(self,audio_tensor, word_indices, codebook_index=0):
        audio_tokens = self.mimi.encode(audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device))
        
        truncated_word_indices = [ idx for idx in word_indices if idx < audio_tokens.shape[2]]
        selected_tokens = audio_tokens[0, codebook_index, truncated_word_indices]

        codeword_table = self._get_codebook_table(codebook_index)

        return codeword_table[selected_tokens].T, truncated_word_indices 

    def _get_codebook_table(self, codebook_index):
        if codebook_index == 0:
            codeword_table = self.mimi.quantizer.rvq_first.vq.layers[0]._codebook._embedding
        else: 
            codeword_table = self.mimi.quantizer.rvq_rest.vq.layers[codebook_index-1]._codebook._embedding
        return codeword_table#Size [256, N]
    
    def get_latent_for_last_word(self,audio_path: Path, embedding_type: str, textgrid_path: Path | None = None, codebook_index=0, latent_rate_hz=12.5):
        if textgrid_path == None: 
            textgrid_path = audio_path.with_suffix(".TextGrid")
        accent_idx = self._get_token_indices_for_word(textgrid_path, latent_rate_hz)

        audio_tensor = self._load_audio(audio_path)

        if embedding_type == "unquantized":
            embs, truncated_idx = self._get_unquantized_latents_for_last_word(audio_tensor=audio_tensor, word_indices=accent_idx)
            if len(truncated_idx) != len(accent_idx):
                print(f"Indices of accented were truncated for {audio_path}: extracted indices were {accent_idx}, truncated indices were {truncated_idx}")
            return embs
        elif embedding_type == "prequantized":
            embs, truncated_idx = self._get_prequantized_latents_for_last_word(audio_tensor=audio_tensor, word_indices=accent_idx)
            if len(truncated_idx) != len(accent_idx):
                print(f"Indices of accented were truncated for {audio_path}: extracted indices were {accent_idx}, truncated indices were {truncated_idx}")
            return embs
        elif embedding_type == "codebook":
            embs, truncated_idx = self._get_codebook_latents_for_last_word(audio_tensor=audio_tensor, word_indices=accent_idx, codebook_index=codebook_index)
        else:
            raise ValueError(f"Unknown embedding type '{embedding_type}'")

        if len(truncated_idx) != len(accent_idx):
            print(f"Indices of accented were truncated for {audio_path}: extracted indices were {accent_idx}, truncated indices were {truncated_idx}")
        return embs

    def _load_audio(self, audio_path):
        waveform, sr = torchaudio.load(str(audio_path))
        audio_tensor = torchaudio.functional.resample(
            waveform.squeeze(0), orig_freq=sr, new_freq=self.mimi.sample_rate
        )
        
        return audio_tensor
    
    def decode_tokens(self, tokens, *, keep_levels: int | None = None):
        """Decode an RVQ token array back to a waveform.

        ``tokens`` is ``(T, K)`` as the generation archives store it, or ``(K, T)``; the
        orientation is inferred, since ``K`` is fixed by the codec while ``T`` varies.

        Passing fewer than the codec's full complement of levels is meaningful and is how
        depth-truncated reconstruction is done: a residual quantizer reconstructs by summing
        codeword vectors level by level, so a prefix of the levels is exactly what a shallower
        codec would hand the decoder.

        Returns a ``(1, samples)`` float tensor at :attr:`sample_rate`.
        """
        import numpy as np

        codes = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
        if codes.dim() != 2:
            raise ValueError(f"expected a 2-D token array, got shape {tuple(codes.shape)}")

        configured = self.mimi.num_codebooks
        if codes.shape[0] == configured and codes.shape[1] != configured:
            codes = codes.T                      # (K, T) supplied; store orientation is (T, K)
        if codes.shape[1] > configured:
            raise ValueError(
                f"{codes.shape[1]} levels supplied but this manager is configured for "
                f"{configured}. Construct it with num_codebooks={codes.shape[1]} to decode "
                "a token stream from a generator that uses more levels than the probing "
                "experiments do.")

        if keep_levels is not None:
            if not 1 <= keep_levels <= codes.shape[1]:
                raise ValueError(f"keep_levels={keep_levels} outside 1..{codes.shape[1]}")
            # A residual quantizer sums codeword vectors level by level, so decoding a prefix
            # is well defined and is exactly what a shallower codec would hand the decoder.
            codes = codes[:, :keep_levels]

        with torch.no_grad():
            return self.mimi.decode(codes.T.unsqueeze(0).to(self.device))[0].cpu()

    def get_latents_for_utterance(self, audio_path: Path, embedding_type: str, codebook_index=0):
        audio_tensor = self._load_audio(audio_path)

        if embedding_type == "unquantized":
            with torch.no_grad():
                return self.mimi.encode_to_latent(audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device))
        elif embedding_type == "prequantized":
            with torch.no_grad():
                return self.mimi.encode_to_latent(
                    audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device), quantize=False)
        elif embedding_type == "codebook":
            audio_tokens = self.mimi.encode(audio_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device))
            codebook_token_ids = audio_tokens[0, codebook_index, :]
            codeword_table = self._get_codebook_table(codebook_index)
            return codeword_table[codebook_token_ids].T # Size [256, D]
        else:
            raise ValueError(f"Unknown embedding type '{embedding_type}'")


class WavLMAudioManager(AudioManager):
    """
    WavLM-Large only supports 'unquantized' embeddings (last_hidden_state).
    It is an encoder-only model with no discrete codebook; 'codebook' will raise ValueError.
    Default latent rate is 50 Hz (16000 Hz / 320 stride). Pass latent_rate_hz=50.0 at call sites.
    """
    def __init__(self, device, layer: int | None = None):
        """layer: which transformer layer's hidden states to read. None (default) uses
        `last_hidden_state` (the final layer, index 24 for WavLM-large). An int L in
        1..24 reads hidden_states[L] (task B9: a middle layer, e.g. 12), so a second WavLM
        representation can be probed alongside the final one."""
        self.device = device
        self.layer = layer
        self.sample_rate = 16000
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
        self.model = AutoModel.from_pretrained(WAVLM_MODEL_ID).to(self.device)
        self.model.eval()

    def _forward_hidden(self, inputs):
        """Return the (1, T, D) hidden states of the configured layer (last layer if
        self.layer is None). hidden_states is a tuple of length num_layers+1 with
        hidden_states[-1] == last_hidden_state, so layer L maps to hidden_states[L]."""
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=(self.layer is not None))
        if self.layer is not None:
            return outputs.hidden_states[self.layer]  # (1, T, D)
        return outputs.last_hidden_state              # (1, T, D)

    def _get_token_indices_for_word(self, textgrid_path: Path, latent_rate_hz=50.0, word_position: int = -1):
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        words_tier = tg.getTier("words")
        start, end, _ = words_tier.entries[word_position]
        t_start = int(np.floor(start * latent_rate_hz))
        t_end = int(np.ceil(end * latent_rate_hz))
        if word_position == -1:
            return np.arange(t_start, t_end + 1)
        else:
            return np.arange(t_start, t_end)

    def _load_audio(self, audio_path: Path):
        waveform, sr = torchaudio.load(str(audio_path))
        return torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sr, new_freq=self.sample_rate)

    def _get_unquantized_latents_for_last_word(self, audio_tensor, word_indices):
        inputs = self.feature_extractor(
            audio_tensor.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden_states = self._forward_hidden(inputs)  # (1, T, D)
        truncated = [idx for idx in word_indices if idx < hidden_states.shape[1]]
        return hidden_states[0, truncated, :].T, truncated  # (D, T)

    def get_latent_for_last_word(self, audio_path: Path, embedding_type: str, textgrid_path: Path | None = None, codebook_index=0, latent_rate_hz=50.0):
        if embedding_type == "codebook":
            raise ValueError("WavLM is encoder-only and has no discrete codebook; use embedding_type='unquantized'")
        if embedding_type != "unquantized":
            raise ValueError(f"Unknown embedding type '{embedding_type}'")

        if textgrid_path is None:
            textgrid_path = audio_path.with_suffix(".TextGrid")
        accent_idx = self._get_token_indices_for_word(textgrid_path, latent_rate_hz)
        audio_tensor = self._load_audio(audio_path)
        embs, truncated_idx = self._get_unquantized_latents_for_last_word(audio_tensor, accent_idx)
        if len(truncated_idx) != len(accent_idx):
            print(f"Indices truncated for {audio_path}: extracted {accent_idx}, truncated {truncated_idx}")
        return embs

    def get_latents_for_utterance(self, audio_path: Path, embedding_type: str, codebook_index=0):
        if embedding_type == "codebook":
            raise ValueError("WavLM is encoder-only and has no discrete codebook; use embedding_type='unquantized'")
        if embedding_type != "unquantized":
            raise ValueError(f"Unknown embedding type '{embedding_type}'")

        audio_tensor = self._load_audio(audio_path)
        inputs = self.feature_extractor(
            audio_tensor.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._forward_hidden(inputs).transpose(1, 2)  # (1, D, T) to match Mimi convention


class SpeechTokenizerAudioManager(AudioManager):
    """
    SpeechTokenizer supports both 'unquantized' (pre-RVQ encoder output) and
    'codebook' (discrete token → codebook embedding lookup) modes.
    Requires a local config.json and checkpoint .pt file.
    """
    def __init__(self, device, config_path: str | None = None, ckpt_path: str | None = None):
        self.device = device
        if config_path is None:
            config_path = str(config.path('models', 'speechtokenizer', 'config'))
        if ckpt_path is None:
            ckpt_path = str(config.path('models', 'speechtokenizer', 'checkpoint'))
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.model.eval()
        self.model.to(self.device)
        self.sample_rate = self.model.sample_rate

    def _get_token_indices_for_word(self, textgrid_path: Path, latent_rate_hz=50.0, word_position: int = -1):
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        words_tier = tg.getTier("words")
        start, end, _ = words_tier.entries[word_position]
        t_start = int(np.floor(start * latent_rate_hz))
        t_end = int(np.ceil(end * latent_rate_hz))
        if word_position == -1:
            return np.arange(t_start, t_end + 1)
        else:
            return np.arange(t_start, t_end)

    def _load_audio(self, audio_path: Path):
        waveform, sr = torchaudio.load(str(audio_path))
        return torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sr, new_freq=self.sample_rate)

    def _get_unquantized_latents_for_last_word(self, audio_tensor, word_indices):
        x = audio_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, T)
        with torch.no_grad():
            e = self.model.encoder(x)  # (1, D, T')
        truncated = [idx for idx in word_indices if idx < e.shape[2]]
        return e[0, :, truncated], truncated  # (D, T)

    def _get_codebook_latents_for_last_word(self, audio_tensor, word_indices, codebook_index=0):
        x = audio_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, T)
        with torch.no_grad():
            codes = self.model.encode(x)  # (n_q, 1, T')
        truncated = [idx for idx in word_indices if idx < codes.shape[2]]
        selected_tokens = codes[codebook_index, 0, truncated]
        codeword_table = self._get_codebook_table(codebook_index)
        return codeword_table[selected_tokens].T, truncated  # (D, T)

    def _get_dequantized_latents_for_last_word(self, audio_tensor, word_indices):
        """The residual-vector-quantization reconstruction: all 8 codebooks summed.

        This is SpeechTokenizer's counterpart of what Mimi's `encode_to_latent` returns by
        default, and it is the object whose gap from the encoder output measures the cost of
        quantization. Unlike Mimi's quantizer, SpeechTokenizer's has no input/output
        projections -- `layer.decode` returns codewords in the encoder's own space -- so the
        reconstruction and the encoder output are directly comparable, without a change of
        basis in between.
        """
        x = audio_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, T)
        with torch.no_grad():
            codes = self.model.encode(x)                    # (n_q, 1, T')
            rec = self.model.quantizer.decode(codes)        # (1, D, T')
        truncated = [idx for idx in word_indices if idx < rec.shape[2]]
        return rec[0, :, truncated], truncated              # (D, T)

    def _get_codebook_table(self, codebook_index):
        return self.model.quantizer.vq.layers[codebook_index].codebook  # (codebook_size, D)

    def get_latent_for_last_word(self, audio_path: Path, embedding_type: str, textgrid_path: Path | None = None, codebook_index=0, latent_rate_hz=50.0):
        if textgrid_path is None:
            textgrid_path = audio_path.with_suffix(".TextGrid")
        accent_idx = self._get_token_indices_for_word(textgrid_path, latent_rate_hz)
        audio_tensor = self._load_audio(audio_path)

        if embedding_type == "unquantized":
            embs, truncated_idx = self._get_unquantized_latents_for_last_word(audio_tensor, accent_idx)
        elif embedding_type == "dequantized":
            embs, truncated_idx = self._get_dequantized_latents_for_last_word(audio_tensor, accent_idx)
        elif embedding_type == "codebook":
            embs, truncated_idx = self._get_codebook_latents_for_last_word(audio_tensor, accent_idx, codebook_index)
        else:
            raise ValueError(f"Unknown embedding type '{embedding_type}'")

        if len(truncated_idx) != len(accent_idx):
            print(f"Indices truncated for {audio_path}: extracted {accent_idx}, truncated {truncated_idx}")
        return embs

    def get_latents_for_utterance(self, audio_path: Path, embedding_type: str, codebook_index=0):
        audio_tensor = self._load_audio(audio_path)
        x = audio_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, T)

        if embedding_type == "unquantized":
            with torch.no_grad():
                return self.model.encoder(x)  # (1, D, T)
        elif embedding_type == "dequantized":
            with torch.no_grad():
                codes = self.model.encode(x)              # (n_q, 1, T)
                return self.model.quantizer.decode(codes)  # (1, D, T)
        elif embedding_type == "codebook":
            with torch.no_grad():
                codes = self.model.encode(x)  # (n_q, 1, T)
            token_ids = codes[codebook_index, 0, :]
            codeword_table = self._get_codebook_table(codebook_index)
            return codeword_table[token_ids].T  # (D, T)
        else:
            raise ValueError(f"Unknown embedding type '{embedding_type}'")




class HuBERTAudioManager(WavLMAudioManager):
    """HuBERT-base, SpeechTokenizer's distillation teacher.

    SpeechTokenizer's checkpoint is `speechtokenizer_hubert_avg`: its first codebook is distilled
    from the *average* of HuBERT-base-ls960's transformer-layer representations, which is what
    `layer=None` returns here. An int L reads `hidden_states[L]` instead, so a single layer can be
    probed for comparison. Everything else -- 16 kHz input, 50 Hz frames, the nuclear-word crop --
    matches WavLM, so only the model and the layer aggregation differ and the class subclasses it.

    D = 768, against WavLM-Large's 1024: HuBERT-base is the smaller model, and the comparison of
    the ST codebook with its teacher is a comparison against that representation, not against a
    capacity-matched one.
    """

    def __init__(self, device, layer: int | None = None):
        self.device = device
        self.layer = layer
        self.sample_rate = 16000
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(HUBERT_MODEL_ID)
        self.model = AutoModel.from_pretrained(HUBERT_MODEL_ID).to(self.device)
        self.model.eval()

    def _forward_hidden(self, inputs):
        """(1, T, D) hidden states: layer `self.layer`, or the mean over all layers if None."""
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        if self.layer is not None:
            return outputs.hidden_states[self.layer]
        # hidden_states[0] is the convolutional feature projection, not a transformer layer; the
        # teacher is the average over the transformer layers, so it is excluded.
        return torch.stack(outputs.hidden_states[1:], dim=0).mean(dim=0)
