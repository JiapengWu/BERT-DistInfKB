import torch
# from pytorch_pretrained_bert import BertModel

from pytorch_transformers import BertModel

# from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import torch.nn as nn
import pdb
from utils.common.util import calc_bilinear


class VanillaBertLanguageModel(nn.Module):
    def __init__(self, config):
        super(VanillaBertLanguageModel, self).__init__()
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        pdb.set_trace()

    def forward(self, context_embedding, candidate_embedding, prediction_positions, can_mask, attention_mask=None, output_all_encoded_layers=True):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # context_embedding: bsz, wlen, hsz
        # candidate_embedding: bsz, csz, emb
        bsz, wlen, hsz = context_embedding.shape
        csz = candidate_embedding.shape[1]
        for i in range(bsz):
            first_half = candidate_embedding[i, :prediction_positions[i], :]
            second = candidate_embedding[i, prediction_positions[i]+1, :]


        encoded_layers = self.context_encoder.encoder(context_embedding,
                                                      extended_attention_mask,
                                                      output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = encoded_layers[-1]
        pooled_output = self.dropout(self.context_encoder.pooler(sequence_output))
        return encoded_layers, pooled_output


class BertLeoModel(nn.Module):
    def __init__(self, config):
        super(BertLeoModel, self).__init__()
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.candidate_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.context_encoder.config.hidden_size
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, config.bilinear_size)

    def forward_with_input_embeddings(self, model, input_embeddings, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * model.config.num_hidden_layers
        encoded_layers = model.encoder(input_embeddings, extended_attention_mask, head_mask)

        sequence_output = encoded_layers[-1]
        pooled_output = self.context_encoder.dropout(self.context_encoder.pooler(sequence_output))
        return pooled_output

    def forward(self, contexts_txt, candidate_txt, prediction_positions, pad_idx):
        # contexts_txt: bsz, slen
        # candidate_txt: bsz, csz, 200
        bsz, slen = contexts_txt.shape; csz = candidate_txt.shape[1]

        candidate_attention_mask = candidate_txt != pad_idx

        # pdb.set_trace()
        candidate_encodings = self.candidate_encoder(
            candidate_txt.view(-1, candidate_txt.shape[-1]),
            attention_mask=candidate_attention_mask.view(-1, candidate_txt.shape[-1]))[1] # bsz * csz, hsz


        hsz = candidate_encodings.shape[-1]
        candidate_encodings = candidate_encodings.view(bsz, csz, hsz) # bsz, csz, hsz

        context_embeddings = self.context_encoder.embeddings(contexts_txt)

        contexts_candidates = []
        for i in range(bsz):
            for j in range(csz):
                context_embeddings[i, prediction_positions[i], :] = candidate_encodings[i, j]
                contexts_candidates.append(context_embeddings[i, :, :].unsqueeze(0))  # append: 1, slen, hsz

        context_embeddings = torch.cat(contexts_candidates, dim=0) # bsz * csz, slen, hsz

        context_attention_mask = contexts_txt != pad_idx # bsz, slen
        context_attention_mask = context_attention_mask.repeat(1, csz).view(bsz * csz, slen) # bsz * csz, slen

        context_encodings = self.forward_with_input_embeddings(
            self.context_encoder, context_embeddings, context_attention_mask) # bsz * csz, hsz

        context_encodings = context_encodings.view(bsz, csz, hsz)
        # pdb.set_trace()
        return calc_bilinear(self.bilinear, context_encodings, candidate_encodings)
