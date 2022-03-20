encoder_postfix_mapping = {"self_attn.in_proj_weight":
                               ["self_attn.fc_q.weight",
                                "self_attn.fc_k.weight",
                                "self_attn.fc_v.weight"],
                           "self_attn.in_proj_bias":
                               ["self_attn.fc_q.bias",
                                "self_attn.fc_k.bias",
                                "self_attn.fc_v.bias"],
                           "self_attn.out_proj.weight": "self_attn.fc_concat.weight",
                           "self_attn.out_proj.bias": "self_attn.fc_concat.bias",
                           "linear1.weight": "ffn.fc_1.weight",
                           "linear1.bias": "ffn.fc_1.bias",
                           "linear2.weight": "ffn.fc_2.weight",
                           "linear2.bias": "ffn.fc_2.bias",
                           "norm1.weight": "norm1.weight",
                           "norm1.bias": "norm1.bias",
                           "norm2.weight": "norm2.weight",
                           "norm2.bias": "norm2.bias",
                           }

decoder_postfix_mapping = {"self_attn.in_proj_weight":
                               ["self_attn.fc_q.weight",
                                "self_attn.fc_k.weight",
                                "self_attn.fc_v.weight"],
                           "self_attn.in_proj_bias":
                               ["self_attn.fc_q.bias",
                                "self_attn.fc_k.bias",
                                "self_attn.fc_v.bias"],

                           "self_attn.out_proj.weight": "self_attn.fc_concat.weight",
                           "self_attn.out_proj.bias": "self_attn.fc_concat.bias",
                           "multihead_attn.in_proj_weight":
                               ["cross_attn.fc_q.weight",
                                "cross_attn.fc_k.weight",
                                "cross_attn.fc_v.weight"],
                           "multihead_attn.in_proj_bias":
                               ["cross_attn.fc_q.bias",
                                "cross_attn.fc_k.bias",
                                "cross_attn.fc_v.bias"],
                           "multihead_attn.out_proj.weight": "cross_attn.fc_concat.weight",
                           "multihead_attn.out_proj.bias": "cross_attn.fc_concat.bias",

                           "linear1.weight": "ffn.fc_1.weight",
                           "linear1.bias": "ffn.fc_1.bias",
                           "linear2.weight": "ffn.fc_2.weight",
                           "linear2.bias": "ffn.fc_2.bias",
                           "norm1.weight": "norm1.weight",
                           "norm1.bias": "norm1.bias",
                           "norm2.weight": "norm2.weight",
                           "norm2.bias": "norm2.bias",
                           "norm3.weight": "norm3.weight",
                           "norm3.bias": "norm3.bias",
                           }

layernorm_mapping = {"norm.weight": "norm.weight",
                     "norm.bias": "norm.bias"}

classifier_mapping = {"generator.weight": "classifier.weight",
                      "generator.bias": "classifier.bias"}

embedding_mapping = {"src_tok_emb.embedding.weight": "src_tok_emb.embedding.weight",
                     "tgt_tok_emb.embedding.weight": "tgt_tok_emb.embedding.weight",
                     "positional_encoding.pos_embedding": "positional_encoding.encoding"}
