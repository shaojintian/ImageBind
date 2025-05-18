import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace # For dummy config

# --- Helper: Positional Encoding (from previous code) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

# --- Helper: Patch Embedding for Images (from previous code) ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x); x = x.flatten(2); x = x.transpose(1, 2)
        return x

# --- Modality Specific Embedders (Text, Image, Audio - simplified for brevity) ---
class TextEmbedder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.scale = math.sqrt(embed_dim)
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(text_tokens) * self.scale

class ImageEmbedder(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(images)

class AudioEmbedder(nn.Module):
    def __init__(self, input_features: int, patch_size: int, embed_dim: int, max_audio_len: int):
        super().__init__()
        self.proj = nn.Linear(input_features * patch_size, embed_dim)
        self.patch_size = patch_size
    def forward(self, audio_frames: torch.Tensor, audio_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S_audio, F_audio = audio_frames.shape
        padding_needed = (self.patch_size - (S_audio % self.patch_size)) % self.patch_size
        if padding_needed > 0:
            padding = torch.zeros(B, padding_needed, F_audio, device=audio_frames.device, dtype=audio_frames.dtype)
            audio_frames = torch.cat([audio_frames, padding], dim=1)
            if audio_mask is not None:
                mask_padding = torch.zeros(B, padding_needed, device=audio_mask.device, dtype=audio_mask.dtype)
                audio_mask = torch.cat([audio_mask, mask_padding], dim=1)
        S_padded = audio_frames.size(1)
        audio_patches = audio_frames.contiguous().view(B, S_padded // self.patch_size, self.patch_size * F_audio)
        embedded_patches = self.proj(audio_patches)
        new_mask = audio_mask[:, ::self.patch_size] if audio_mask is not None else None
        return embedded_patches, new_mask


# --- UnifiedEncoder (largely same as before) ---
class UnifiedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.modalities = config.modalities
        self.modality_to_idx = {name: i for i, name in enumerate(self.modalities)}

        self.modality_embedders = nn.ModuleDict()
        self.modality_max_tokens = {} # Max tokens *per modality after patching*

        if "text" in self.modalities:
            text_cfg = config.text_config
            self.modality_embedders["text"] = TextEmbedder(
                text_cfg.vocab_size, self.embed_dim, text_cfg.get("padding_idx", None)
            )
            self.modality_max_tokens["text"] = text_cfg.max_len
        if "image" in self.modalities:
            img_cfg = config.image_config
            self.modality_embedders["image"] = ImageEmbedder(
                img_cfg.img_size, img_cfg.patch_size, img_cfg.in_chans, self.embed_dim
            )
            self.modality_max_tokens["image"] = (img_cfg.img_size // img_cfg.patch_size) ** 2
        if "audio" in self.modalities:
            audio_cfg = config.audio_config
            self.modality_embedders["audio"] = AudioEmbedder(
                audio_cfg.input_features, audio_cfg.patch_size, self.embed_dim, audio_cfg.max_audio_len
            )
            self.modality_max_tokens["audio"] = (audio_cfg.max_audio_len + audio_cfg.patch_size - 1) // audio_cfg.patch_size

        self.cls_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.embed_dim)) for modality in self.modalities
        })
        self.modality_type_embeddings = nn.Embedding(len(self.modalities), self.embed_dim)
        
        # Max possible length for positional encoding (sum of all configured modalities' max tokens + CLS tokens)
        max_possible_encoder_len = sum(
            self.modality_max_tokens.get(mod, 0) + 1 for mod in self.modalities
        )
        self.positional_encoding = PositionalEncoding(self.embed_dim, config.dropout, max_possible_encoder_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size, dropout=config.dropout,
            activation=config.hidden_act, batch_first=True, norm_first=config.get("norm_first", True)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.output_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], attention_masks: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = next(iter(inputs.values())).size(0)
        all_embeddings: List[torch.Tensor] = []
        all_padding_masks: List[torch.Tensor] = [] # True for valid, False for padding

        # Iterate in the pre-defined order of self.modalities to ensure consistent concatenation order
        active_modalities_in_input_order = [mod for mod in self.modalities if mod in inputs]

        for modality_name in active_modalities_in_input_order:
            data = inputs[modality_name]
            modality_idx = self.modality_to_idx[modality_name]
            cls_token = self.cls_embeddings[modality_name].expand(batch_size, -1, -1)
            padding_mask_modality_tokens = None # Mask for tokens part (excluding CLS)

            if modality_name == "text":
                x = self.modality_embedders["text"](data) # (B, S_text, E)
                # Use provided mask or assume all valid
                padding_mask_modality_tokens = (attention_masks.get("text", torch.ones_like(data, dtype=torch.bool, device=data.device))
                                         if attention_masks else torch.ones_like(data, dtype=torch.bool, device=data.device))
            elif modality_name == "image":
                x = self.modality_embedders["image"](data) # (B, N_patches, E)
                padding_mask_modality_tokens = torch.ones(batch_size, x.size(1), dtype=torch.bool, device=x.device)
            elif modality_name == "audio":
                audio_input_mask = attention_masks.get("audio_raw", None) if attention_masks else None
                x, patched_audio_mask = self.modality_embedders["audio"](data, audio_input_mask)
                padding_mask_modality_tokens = patched_audio_mask if patched_audio_mask is not None else torch.ones(batch_size, x.size(1), dtype=torch.bool, device=x.device)
            else:
                raise ValueError(f"Unsupported modality: {modality_name}")

            embeddings_with_cls = torch.cat([cls_token, x], dim=1) # (B, 1 + S_mod, E)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=embeddings_with_cls.device) # CLS is always valid
            current_modality_padding_mask = torch.cat([cls_mask, padding_mask_modality_tokens], dim=1) # (B, 1 + S_mod)
            
            mod_type_emb = self.modality_type_embeddings(torch.tensor([modality_idx], device=embeddings_with_cls.device))
            embeddings_with_cls += mod_type_emb.unsqueeze(0) # Add modality type
            
            all_embeddings.append(embeddings_with_cls)
            all_padding_masks.append(current_modality_padding_mask)

        if not all_embeddings:
            # This case should ideally be handled by checking inputs before calling encoder
            # Or return a specific zero tensor if that's meaningful for some downstream tasks
            raise ValueError("No modalities provided to encoder for processing.")

        concatenated_embeddings = torch.cat(all_embeddings, dim=1)
        combined_padding_mask = torch.cat(all_padding_masks, dim=1) # True for valid tokens
        
        # Apply positional encoding to the fully concatenated sequence
        concatenated_embeddings = self.positional_encoding(concatenated_embeddings[:, :self.positional_encoding.pe.size(0)]) # Slice PE if seq too long
        
        # Transformer expects padding mask where True means "masked" / "ignore"
        transformer_src_key_padding_mask = ~combined_padding_mask

        encoder_outputs = self.transformer_encoder(
            concatenated_embeddings,
            src_key_padding_mask=transformer_src_key_padding_mask
        )
        encoder_outputs = self.output_norm(encoder_outputs)
        
        # Return combined_padding_mask (True for valid) as it's useful for decoder's memory_key_padding_mask
        return encoder_outputs, combined_padding_mask


# --- OmniModel using the UnifiedEncoder and adding a Decoder ---
class OmniModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.encoder_config.embed_dim # Common embedding dimension

        self.encoder = self._get_encoder(config.encoder_config)
        
        # Decoder components (assuming text generation as primary target for 'generate')
        self.decoder = None
        if hasattr(config, "decoder_config") and config.decoder_config is not None:
            self.decoder = self._get_decoder(config.decoder_config, self.embed_dim)
            
            # For text generation, we need specific components
            if "text" in config.decoder_config.target_modalities:
                # Use text_config from encoder_config for vocab_size, padding_idx etc.
                text_cfg_ref = config.encoder_config.text_config 
                
                self.tgt_text_embedder = TextEmbedder(
                    text_cfg_ref.vocab_size, self.embed_dim, text_cfg_ref.padding_idx
                )
                # Max length for generated text sequence
                decoder_max_len = getattr(config.decoder_config, "max_len", text_cfg_ref.max_len)
                self.tgt_text_pos_encoder = PositionalEncoding(
                    self.embed_dim, config.decoder_config.dropout, decoder_max_len
                )
                self.text_output_projection = nn.Linear(self.embed_dim, text_cfg_ref.vocab_size)

                # Option to tie weights of target embedding and output projection
                if config.decoder_config.get("tie_output_projection_weights", False):
                    self.text_output_projection.weight = self.tgt_text_embedder.embedding.weight
        else:
            # If no decoder config, model might be used for embedding tasks only
            pass


    def _get_encoder(self, encoder_config):
        return UnifiedEncoder(encoder_config)

    def _get_decoder(self, decoder_config, embed_dim):
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=decoder_config.num_attention_heads,
            dim_feedforward=decoder_config.intermediate_size,
            dropout=decoder_config.dropout,
            activation=decoder_config.hidden_act,
            batch_first=True,
            norm_first=decoder_config.get("norm_first", True)
        )
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_config.num_hidden_layers)
        return transformer_decoder

    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                attention_masks: Optional[Dict[str, torch.Tensor]] = None,
                decoder_input_ids: Optional[torch.Tensor] = None, # e.g., for text: (B, S_tgt)
                decoder_attention_mask: Optional[torch.Tensor] = None # (B, S_tgt), True for valid
               ):
        """
        Full forward pass for training or inference with teacher forcing.
        """
        encoder_hidden_states, encoder_combined_padding_mask = self.encoder(inputs, attention_masks)
        # encoder_combined_padding_mask is True for VALID tokens.
        # nn.TransformerDecoder memory_key_padding_mask needs True for MASKED tokens.
        memory_key_padding_mask = ~encoder_combined_padding_mask if encoder_combined_padding_mask is not None else None

        if self.decoder is None or decoder_input_ids is None:
            # Encoder-only usage or if decoder inputs are not provided
            return encoder_hidden_states, encoder_combined_padding_mask

        # --- Decoder Pass (Assuming Text Generation Target) ---
        if not hasattr(self, 'tgt_text_embedder'):
            raise RuntimeError("Decoder target embedder (tgt_text_embedder) not initialized. Check decoder config.")

        tgt_embeddings = self.tgt_text_embedder(decoder_input_ids)
        tgt_embeddings = self.tgt_text_pos_encoder(tgt_embeddings)

        # Causal mask for decoder self-attention (S_tgt, S_tgt)
        tgt_seq_len = decoder_input_ids.size(1)
        causal_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=decoder_input_ids.device)

        # Padding mask for target sequence (if any)
        # decoder_attention_mask is True for VALID. nn.TransformerDecoderLayer needs True for MASKED.
        tgt_key_padding_mask = ~decoder_attention_mask if decoder_attention_mask is not None else None
        
        decoder_output = self.decoder(
            tgt=tgt_embeddings,
            memory=encoder_hidden_states,
            tgt_mask=causal_tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        logits = self.text_output_projection(decoder_output) # (B, S_tgt, V_text)
        
        # Return logits and also encoder outputs if needed for other tasks or analysis
        return logits, encoder_hidden_states, encoder_combined_padding_mask


    @torch.no_grad()
    def generate(self,
                 inputs: Dict[str, torch.Tensor], # Inputs for the encoder
                 attention_masks: Optional[Dict[str, torch.Tensor]] = None,
                 target_modality: str = "text",
                 max_length: int = 50,
                 sos_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 pad_token_id: Optional[int] = 0, # Default to common padding id
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 # num_beams: int = 1 # For beam search (more complex)
                ):
        """
        Autoregressive generation for the specified target_modality (currently 'text').
        Uses greedy search or sampling with temperature, top-k, top-p.
        """
        self.eval() # Ensure model is in evaluation mode

        if self.decoder is None or not hasattr(self, f"tgt_{target_modality}_embedder"):
            raise RuntimeError(f"Decoder or target {target_modality} embedder not configured for generation.")
        if target_modality != "text":
            raise NotImplementedError(f"Generation for target_modality '{target_modality}' is not implemented yet.")
        
        # Use text_config from encoder for token IDs if available
        text_cfg = self.config.encoder_config.text_config
        _sos_token_id = sos_token_id if sos_token_id is not None else getattr(text_cfg, "sos_token_id", None)
        _eos_token_id = eos_token_id if eos_token_id is not None else getattr(text_cfg, "eos_token_id", None)
        _pad_token_id = pad_token_id if pad_token_id is not None else getattr(text_cfg, "padding_idx", 0)

        if _sos_token_id is None:
            # If no specific SOS, often models start with PAD or a learned start token.
            # For simplicity, we'll require it or use PAD.
            _sos_token_id = _pad_token_id 
            print(f"Warning: sos_token_id not provided, using pad_token_id ({_sos_token_id}) as start token.")


        batch_size = next(iter(inputs.values())).size(0)
        device = next(self.parameters()).device

        # 1. Encode input modalities
        encoder_hidden_states, encoder_combined_padding_mask = self.encoder(inputs, attention_masks)
        memory_key_padding_mask = ~encoder_combined_padding_mask if encoder_combined_padding_mask is not None else None

        # 2. Initialize generated sequence for each item in the batch
        # Start with SOS token: (B, 1)
        generated_ids = torch.full((batch_size, 1), _sos_token_id, dtype=torch.long, device=device)
        
        # Keep track of which sequences are finished
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1): # -1 because SOS is already the first token
            if is_finished.all(): # Stop if all sequences in batch are done
                break

            # Prepare decoder inputs from the current generated_ids
            # (B, current_seq_len) -> (B, current_seq_len, E)
            current_tgt_embeddings = self.tgt_text_embedder(generated_ids)
            current_tgt_embeddings = self.tgt_text_pos_encoder(current_tgt_embeddings)

            # Causal mask for decoder self-attention
            tgt_seq_len = generated_ids.size(1)
            causal_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
            
            # Get decoder output for the current step
            # We pass the full generated sequence so far.
            # The nn.TransformerDecoder handles attending only to previous tokens via causal_tgt_mask.
            decoder_output_step = self.decoder(
                tgt=current_tgt_embeddings,       # (B, current_seq_len, E)
                memory=encoder_hidden_states,     # (B, S_encoder, E)
                tgt_mask=causal_tgt_mask,         # (current_seq_len, current_seq_len)
                memory_key_padding_mask=memory_key_padding_mask # (B, S_encoder)
            ) # (B, current_seq_len, E)

            # Get logits for the *next* token (from the last position of decoder_output_step)
            next_token_logits = self.text_output_projection(decoder_output_step[:, -1, :]) # (B, V_text)

            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                # Set all logits not in top-k to -infinity
                indices_to_remove = next_token_logits < top_k_values[:, [-1]]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 # Always keep the most probable token

                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample next token (greedy or multinomial)
            probs = F.softmax(next_token_logits, dim=-1) # (B, V_text)
            # next_token_ids = torch.multinomial(probs, num_samples=1) # (B, 1) # For sampling
            next_token_ids = torch.argmax(probs, dim=-1, keepdim=True) # (B,1) # For greedy

            # If a sequence was finished, keep its EOS token, otherwise append the new token
            if _eos_token_id is not None:
                 next_token_ids[is_finished] = _pad_token_id # Pad finished sequences
            
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

            # Update finished status if EOS is generated and sequence is not already finished
            if _eos_token_id is not None:
                is_finished = is_finished | (next_token_ids.squeeze(-1) == _eos_token_id)
        
        return generated_ids