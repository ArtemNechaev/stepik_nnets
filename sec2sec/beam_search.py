from typing import Sequence
import torch


def beam_search_rnn( model, src, beam_size=5, max_len=20, device=None):
  """
  src - src_len x batch_size

  return tensor max_len x batch_size x beam_size
  """
  model.eval()
  if device == None:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

  src = src.to(model.device)
  batch_size = src.shape[1]

  # src_len x batch_size x hidden_size
  encoder_outputs, h_0 = model.encode(src) 

  # 1 x batch_size
  input_trg = src[0].unsqueeze(0) 
  mask = model.create_pad_mask(src, input_trg)

  # 1 x batch_size x vocab_size
  first_decode, att, h_0 = model.decode(encoder_outputs, input_trg, mask, h_0=h_0 ) 
  first_decode = first_decode.log_softmax(-1)

  # 1 x batch_size x beam_size
  scores, indeces = first_decode.topk(beam_size) 

  # decoder hidden_state. num_layers * D, batch_size*beam_size, hidden
  h_0 = h_0.unsqueeze(2).repeat(1,1,beam_size,1).reshape(h_0.shape[0], batch_size*beam_size, -1) 

  # src_len x batch_size*beam_size x hidden_size
  encoder_outputs = encoder_outputs.unsqueeze(2).repeat(1,1,beam_size,1).reshape(src.shape[0], batch_size*beam_size, -1)

  # src_len x batch_size*beam_size
  src = src.unsqueeze(2).repeat(1,1,beam_size).reshape(src.shape[0], batch_size*beam_size)

  for i in range(1, max_len):
    input_trg = indeces[-1].view(1, batch_size * beam_size )
    mask = model.create_pad_mask(src, input_trg)
    dec_out , att, h_0 = model.decode(encoder_outputs, input_trg, mask, h_0=h_0)

    # 1 x batch_size*beam_size x vocab_size
    new_score = dec_out.log_softmax(-1) 

    # 1 x batch_size*beam_size x beam_size
    new_score, new_idx = new_score.topk(beam_size)

    new_score[:, torch.where(indeces[i-1].flatten() == model.eos_idx)[0]] = 0
    scores = scores.reshape(1, batch_size*beam_size, 1) + new_score

    lens_hyp = ((indeces != model.eos_idx) & (indeces != model.pad_idx)).sum(0, keepdim=True).reshape(1, batch_size*beam_size, 1)
    scores = scores/(lens_hyp+1)**0.5

    ######scores, new_idx = scores.topk(beam_size) #1 x  batch_size*beam_size x beam_size
    flatten_scores = scores.reshape(1,batch_size, beam_size ** 2)

    # 1 x batch_size x beam_size. order_scores [0 .. beam_size**2 - 1]
    scores, order_scores = flatten_scores.topk(beam_size) 

    
    # calc beam indexes that got top beam_size scores. beam_ids [0 .. beam_size - 1]
    beam_ids = torch.div(order_scores, beam_size, rounding_mode='floor') 

    #select beams with top scores from indeces and concat new idx.
    #start of sentences could repeat.

    indeces = torch.cat([
                         torch.gather(indeces, -1, beam_ids.repeat(indeces.shape[0], 1, 1)),
                         torch.gather(new_idx.view(1, batch_size, -1),-1,order_scores)
    ])

    #select beams with top scores from decoder hidden_state
    h_0 = h_0.reshape(h_0.shape[0], batch_size, beam_size, -1)
    h_0 = torch.gather(h_0, 2, beam_ids.unsqueeze(3).repeat(h_0.shape[0], 1, 1, h_0.shape[3]))
    h_0 = h_0.reshape(h_0.shape[0], batch_size*beam_size, -1)


  return indeces, scores
