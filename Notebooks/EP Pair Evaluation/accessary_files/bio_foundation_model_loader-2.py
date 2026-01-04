import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
import torch.nn.functional as F
import sys
import transformers.models.bert.modeling_bert
import builtins

class BlockImport:
    def __init__(self, *blocked):
        self.blocked = set(blocked)

    def __enter__(self):
        self._orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if any(name == b or name.startswith(b + ".") for b in self.blocked):
                raise ImportError(f"Blocked import of {name}")
            return self._orig_import(name, *args, **kwargs)

        builtins.__import__ = fake_import

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.__import__ = self._orig_import


class dnalm_embedding_extraction():
    def __init__(self, model_class, model_name, device):
        self.model_class = model_class
        if model_class=="DNABERT2":
            self.model_name = f"zhihan1996/{model_name}"
            # with NoModule("triton"):
            # with NoTriton():
            with BlockImport("triton"):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                config = BertConfig.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, config=config, trust_remote_code=True)
                self.mask_token = self.tokenizer.mask_token_id
        elif model_class=="HyenaDNA":
            self.model_name = f"LongSafari/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

        elif model_class=="Nucleotide Transformer":
            self.model_name = f"InstaDeepAI/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
            self.mask_token = self.tokenizer.mask_token_id
        elif model_class=="Caduceus":
            self.model_name = f"kuleshov-group/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
            self.mask_token = self.tokenizer.mask_token_id
        elif model_class=="Mistral":
            self.model_name = f"RaphaelMourad/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            self.mask_token = self.tokenizer.mask_token_id
        elif model_class=="GENA-LM":
            self.model_name = f"AIRI-Institute/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.mask_token = self.tokenizer.mask_token_id
        else:
          print("Model not supported.")
        self.device = device
        self.model.to(self.device)
        self.model.eval()


    @property
    def start_token(self):
        if self.model_class=="HyenaDNA":
            return None
        elif self.model_class=="DNABERT2":
            return 1
        elif self.model_class=="Nucleotide Transformer":
            return 3
        elif self.model_class=="Caduceus":
            return None
        elif self.model_class=="Mistral":
            return 1
        elif self.model_class=="GENA-LM":
            return 1

    @property
    def end_token(self):
        if self.model_class=="HyenaDNA":
            return 1
        elif self.model_class=="DNABERT2":
            return 2
        elif self.model_class=="Nucleotide Transformer":
            return None
        elif self.model_class=="Caduceus":
            return 1
        elif self.model_class=="Mistral":
            return 2
        elif self.model_class=="GENA-LM":
            return 2

    def get_embedding(self, sequences, batch_size):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            # if i%50000==0:
            #     print(i)
            batch = sequences[i:min(i+batch_size, len(sequences))]

            if self.model_class=="Nucleotide Transformer":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                hidden = torch_outs.hidden_states[-1]
                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            elif self.model_class=="Mistral":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                hidden = torch_outs.hidden_states[-1]

                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            elif self.model_class=="HyenaDNA":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                hidden = torch_outs.hidden_states[-1]

                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            elif self.model_class=="DNABERT2":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                # !!! due to the bug in its code, DNABERT2 can only return the last hidden layer
                hidden = torch_outs.hidden_states
                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            elif self.model_class=="Caduceus":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                hidden = torch_outs.hidden_states[-1]
                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            elif self.model_class=="GENA-LM":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                clip_mask = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(clip_mask.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))
                if attention_mask is not None:
                    clip_mask = clip_mask * attention_mask

                hidden = torch_outs.hidden_states[-1]
                mask = clip_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts

            embeddings.append(mean_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def get_embedding_weighted(self, sequences, dnases, batch_size):
        embeddings = []
    
        def _compute_starts_ends(tokens, attention_mask):
            B, L = tokens.shape
            device = tokens.device
    
            if self.start_token is not None:
                start_hits = (tokens == self.start_token)
                starts = start_hits.float().argmax(dim=1) + 1
                starts = torch.where(start_hits.any(dim=1), starts, torch.zeros(B, device=device, dtype=torch.long))
            else:
                starts = torch.zeros(B, device=device, dtype=torch.long)
    
            if self.end_token is not None:
                end_hits = (tokens == self.end_token)
                ends = end_hits.float().argmax(dim=1)
                fallback = attention_mask.sum(dim=1) if attention_mask is not None else torch.full((B,), L, device=device, dtype=torch.long)
                ends = torch.where(end_hits.any(dim=1), ends, fallback)
            else:
                ends = attention_mask.sum(dim=1) if attention_mask is not None else torch.full((B,), L, device=device, dtype=torch.long)
    
            return starts, ends
    
        def _clip_mask_from_starts_ends(tokens, attention_mask, starts, ends, dtype):
            B, L = tokens.shape
            pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
            clip_mask = ((pos >= starts.unsqueeze(1)) & (pos < ends.unsqueeze(1))).to(dtype)
            if attention_mask is not None:
                clip_mask = clip_mask * attention_mask.to(dtype)
            return clip_mask
    
        def _dnase_to_token_weights(dnase_batch, tokens, starts, ends, dtype):
            B, L = tokens.shape
            w = torch.zeros((B, L), device=tokens.device, dtype=dtype)
            for b in range(B):
                s = int(starts[b].item())
                e = int(ends[b].item())
                clip_len = max(e - s, 0)
                if clip_len == 0:
                    continue
    
                vec = dnase_batch[b]
                if isinstance(vec, np.ndarray):
                    vec_t = torch.from_numpy(vec)
                else:
                    vec_t = torch.as_tensor(vec)
                vec_t = vec_t.to(device=tokens.device, dtype=dtype).flatten()
    
                if vec_t.numel() == L:
                    w[b, :] = vec_t
                elif vec_t.numel() == clip_len:
                    w[b, s:e] = vec_t
                else:
                    v = vec_t.view(1, 1, -1)
                    v_rs = F.interpolate(v, size=clip_len, mode="linear", align_corners=False).view(-1)
                    w[b, s:e] = v_rs
    
            return w.clamp(min=0.0)
    
        def _weighted_pool(hidden, clip_mask, dnase_w):
            w = (dnase_w * clip_mask).unsqueeze(-1)
            summed = (hidden * w).sum(dim=1)
            denom = w.sum(dim=1).clamp(min=1e-9)
            return summed / denom
    
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : min(start + batch_size, len(sequences))]
            dnase_batch = dnases[start : min(start + batch_size, len(dnases))]
    
            if self.model_class == "GENA-LM":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
    
            tokens = encoded["input_ids"].to(device=self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=self.device)
    
            with torch.no_grad():
                if self.model_class == "Mistral" or self.model_class == "HyenaDNA":
                    torch_outs = self.model(tokens, output_hidden_states=True, return_dict=True)
                else:
                    torch_outs = self.model(tokens, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    
            if self.model_class == "DNABERT2":
                hidden = torch_outs.hidden_states
            else:
                hidden = torch_outs.hidden_states[-1]
    
            starts, ends = _compute_starts_ends(tokens, attention_mask)
            clip_mask = _clip_mask_from_starts_ends(tokens, attention_mask, starts, ends, hidden.dtype)
            dnase_w = _dnase_to_token_weights(dnase_batch, tokens, starts, ends, hidden.dtype)
    
            mean_embeddings = _weighted_pool(hidden, clip_mask, dnase_w)
            embeddings.append(mean_embeddings.cpu().numpy())
    
        return np.vstack(embeddings)



    def get_likelihood(self, sequences, batch_size):
        """
        Compute log-likelihoods of sequences.
        Returns: numpy array of log-likelihoods (one per sequence)
        """
        import torch.nn.functional as F

        likelihoods = []

        for i in range(0, len(sequences), batch_size):
            # if i % 50000 == 0:
            #     print(i)
            batch = sequences[i:min(i+batch_size, len(sequences))]

            if self.model_class == "Nucleotide Transformer":

                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)
                lls = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(tokens.shape[1]):
                    clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
                    masked_tokens = tokens.clone()
                    masked_tokens[:,i,...] = self.mask_token
                    with torch.no_grad():
                        torch_outs = self.model(
                            masked_tokens,
                            attention_mask=attention_mask,
                        )
                        logits = torch_outs.logits.swapaxes(1, 2)
                        tmp = -F.cross_entropy(logits, tokens, reduction="none")
                        lls[:,i] = tmp[:,i] * clip_mask

                seq_likelihoods = lls.sum(dim=1).numpy(force=True)

            elif self.model_class == "Mistral":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                    )
                    logits = torch_outs.logits.swapaxes(1, 2)
                    lls = torch.zeros(tokens.shape[:2], device=self.device)
                    lls[:,1:] = -F.cross_entropy(logits[:,:,:-1], tokens[:,1:], reduction="none")

                clip_mask = torch.zeros_like(lls)
                for i in range(lls.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))

                seq_likelihoods = (lls * clip_mask).sum(1).numpy(force=True)

            elif self.model_class == "HyenaDNA":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                with torch.no_grad():
                    torch_outs = self.model(
                        tokens,
                    )
                    logits = torch_outs.logits.swapaxes(1, 2)
                    lls = torch.zeros(tokens.shape[:2], device=self.device)
                    lls[:,1:] = -F.cross_entropy(logits[:,:,:-1], tokens[:,1:], reduction="none")

                clip_mask = torch.zeros_like(lls)
                for i in range(lls.shape[1]):
                    clip_mask[:,i] = ((i >= starts) & (i < ends))

                seq_likelihoods = (lls * clip_mask).sum(1).numpy(force=True)

            elif self.model_class == "DNABERT2":

                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)
                lls = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(tokens.shape[1]):
                    clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
                    masked_tokens = tokens.clone()
                    masked_tokens[:,i,...] = self.mask_token
                    with torch.no_grad():
                        torch_outs = self.model(
                            masked_tokens,
                            attention_mask=attention_mask,
                        )
                        logits = torch_outs.logits.swapaxes(1, 2)
                        tmp = -F.cross_entropy(logits, tokens, reduction="none")
                        lls[:,i] = tmp[:,i] * clip_mask

                seq_likelihoods = lls.sum(dim=1).numpy(force=True)

            elif self.model_class == "Caduceus":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)
                lls = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(tokens.shape[1]):
                    clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
                    masked_tokens = tokens.clone()
                    masked_tokens[:,i,...] = self.mask_token
                    with torch.no_grad():
                        torch_outs = self.model(
                            masked_tokens,
                            attention_mask=attention_mask,
                        )
                        logits = torch_outs.logits.swapaxes(1, 2)
                        tmp = -F.cross_entropy(logits, tokens, reduction="none")
                        lls[:,i] = tmp[:,i] * clip_mask

                seq_likelihoods = lls.sum(dim=1).numpy(force=True)
            elif self.model_class == "GENA-LM":
                encoded = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if self.start_token is not None:
                    starts = torch.where(tokens == self.start_token)[1] + 1
                else:
                    starts = 0
                if self.end_token is not None:
                    ends = torch.where(tokens == self.end_token)[1]
                else:
                    ends = attention_mask.sum(dim=1)

                tokens = tokens.to(device=self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)
                lls = torch.zeros(tokens.shape[:2], device=self.device)
                for i in range(tokens.shape[1]):
                    clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
                    masked_tokens = tokens.clone()
                    masked_tokens[:,i,...] = self.mask_token
                    with torch.no_grad():
                        torch_outs = self.model(
                            masked_tokens,
                            attention_mask=attention_mask,
                        )
                        logits = torch_outs.logits.swapaxes(1, 2)
                        tmp = -F.cross_entropy(logits, tokens, reduction="none")
                        lls[:,i] = tmp[:,i] * clip_mask

                seq_likelihoods = lls.sum(dim=1).numpy(force=True)

            likelihoods.append(seq_likelihoods)

        return np.concatenate(likelihoods)