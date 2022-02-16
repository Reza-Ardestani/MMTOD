from flask import Flask, request, jsonify, render_template, session
import datetime
import time
import random
import logging

##__________________________________ GPT-3 code  __________________________________________##
from colorama import Fore, Back, Style
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import sys, os
import pprint
import numpy as np
import torch
from image_handler import Handler
# args = ArgsParser().parse()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

img_handler_obj = Handler()
pp = pprint.PrettyPrinter(indent=4)
prev_beliefs = {}
domain_queue = []
# sys.stdout.flush()
model_checkpoint = "./output/checkpoint-108420"
decoding = "DECODING METHOD HERE"
## if decoding == 'nucleus':
##     TOP_P = float(sys.argv[3])
delay = 0.5
context = ''
input_text = ''
turn = 0
end_of_chat = 0
survey_result = ''

print('\nLoading Model', end="")

if 'openai' in model_checkpoint:
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
    model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

# model.load_state_dict(torch.load(model_checkpoint))
model.eval()
model.to('cpu')

break_tokens = tokenizer.encode(tokenizer.eos_token) + tokenizer.encode('?') + tokenizer.encode('!')
# break_tokens = tokenizer.encode(tokenizer.eos_token)
MAX_LEN = model.config.n_ctx

if 'openai-gpt' in model_checkpoint:
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})


def get_belief_new_dbsearch(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|endofbelief|>')[0]
    # elif 'belief.' in sent:
    #     tmp = sent.strip(' ').split('<belief>')[-1].split('<action>')[0]
    # elif 'belief' not in sent:
    #     return []
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def convert_belief(belief):
    dic = {}
    for bs in belief:
        if bs in [' ', '']:
            continue
        domain = bs.split(' ')[0]
        slot = bs.split(' ')[1]
        if slot == 'book':
            slot = ' '.join(bs.split(' ')[1:3])
            value = ' '.join(bs.split(' ')[3:])
        else:
            value = ' '.join(bs.split(' ')[2:])
        if domain not in dic:
            dic[domain] = {}
        try:
            dic[domain][slot] = value
        except:
            print(domain)
            print(slot)
    return dic

def get_turn_domain(beliefs, q):
  for k in beliefs.keys():
      if k not in q:
          q.append(k)
          turn_domain = k
          return turn_domain
  return q[-1]





def get_action_new(sent):
    if '<|action|>' not in sent:
        return []
    elif '<|belief|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|response|>')[0].split('<|action|>')[-1].strip()
    elif '<|action|>' in sent:
        tmp = sent.split('<|response|>')[0].split('<|action|>')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
    # if not tmp.endswith('<endofaction>'):
    #     ipdb.set_trace()
    tmp = tmp.replace('<|endofaction|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    action = tmp.split(',')
    new_action = []
    for act in action:
        if act == '':
            continue
        act = act.strip(' .,')
        if act not in new_action:
            new_action.append(act)
    return new_action



def get_response_new(sent, venuename):
    if '<|response|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|action|>')[-1].split('<|response|>')[-1]
    else:
        return ''
    # if '<belief>' in sent:
    #     tmp = sent.split('<belief>')[-1].split('<action>')[-1].split('<response>')[-1]
    # elif '<action>' in sent:
    #     tmp = sent.split('<action>')[-1].split('<response>')[-1]
    # elif '<response>' in sent:
    #     tmp = sent.split('<response>')[-1]
    # else:
    #     tmp = sent
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofresponse>')
    tmp = tmp.replace('<|endofresponse|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        # if tok in break_tokens:
        if tok in tokenizer.encode(tokenizer.eos_token):
            continue
        new_tokens.append(tok)
    # ipdb.set_trace()
    response = tokenizer.decode(new_tokens).strip(' ,.')
    response = response.replace('[venuename]', '{}'.format(venuename))
    return response


def get_venuename(bs):
    name = ''
    if 'venuename' in bs[0]:
        tmp_list = bs[0].split('venuename')[-1].split(' ')
        #action = tmp_list[-1]
        name = ' '. join(tmp_list[:-1])
    return name


def get_open_span(bs):
    action_names = []
    for tmp in bs[0].split(';'):
        if 'open span' in tmp:
            action = tmp.split('open span')[-1].split(' ')[-1]
            name = tmp.split('open span')[-1].split(action)[0]
            action_names.append((name, action))
    return action_names


##____________________________ End of GPT-3 code __________________________________________##


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'MY_SECRET_KEY'

def Create_message(message):
    global context
    global turn
    global end_of_chat
    logging.warning('In create message')
    global result
    label = session['label']
    state = session['state']
    result = session['result']
    result['turn_id'] = str(turn)
    result['Generated_belief'] = ''
    result['Generated_action'] = ''
    result['response'] = ''
    result['status'] = 'on'
    result['has_image'] = 'False'

    raw_text = message
    input_text = raw_text.replace('you> ', '')
    if end_of_chat > 0:
        end_of_chat +=1
        print(f'value of end_of_chat is {end_of_chat}')
        return ''
    if input_text in ['q', 'quit']:
        print(f'You entered quit and end_of_chat:{end_of_chat}')
        end_of_chat +=1
        return ''

    user = '<|user|> {}'.format(input_text)
    context = context + ' ' + user
    text = '<|endoftext|> <|context|> {} <|endofcontext|>'.format(context)

    # print(context)

    text = text.strip()
    indexed_tokens = tokenizer.encode(text)

    if len(indexed_tokens) > MAX_LEN:
        indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cpu')
    predicted_index = indexed_tokens[-1]

    with torch.no_grad():
        # Greedy decoding

        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            indexed_tokens += [predicted_index]
            tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')
            if len(indexed_tokens) > MAX_LEN:
                break
            if tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                break

    tmp_pred = tokenizer.decode(indexed_tokens)

    print('\ntmp_pred:\n', tmp_pred)

    belief_text = get_belief_new_dbsearch(tmp_pred)
    print('\nbelief_text:\n', belief_text)

    beliefs = convert_belief(belief_text)
    # domain = list(beliefs.keys())[0]
    domain = get_turn_domain(beliefs, domain_queue)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cpu')
    predicted_index = indexed_tokens[-1]

    truncate_action = False
    # Predict all tokens
    with torch.no_grad():
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            indexed_tokens += [predicted_index]
            if len(indexed_tokens) > MAX_LEN:
                break

            predicted_text = tokenizer.decode(indexed_tokens)
            if '<|action|>' in predicted_text:
                generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[0].split(',')
                new_actions = []
                for a in generated_actions:
                    if a in ['', ' ']:
                        continue
                    new_actions.append(a.strip())
                len_actions = len(new_actions)
                if len(list(set(new_actions))) > len(new_actions) or (len_actions > 10 and not truncate_action):
                    # ipdb.set_trace()
                    actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                    indexed_tokens = tokenizer.encode('{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                    # print('action truncated')
                    truncate_action = True
            tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')

    predicted_text = tokenizer.decode(indexed_tokens)
    print('\npredicted_text:\n', predicted_text)

    action_text = get_action_new(predicted_text)
    print('\naction_text:\n', action_text)

    venuename = get_venuename(action_text)
    #print('\nVenuename:\n', venuename)

    response_text = get_response_new(predicted_text, venuename)
    print('\nresponse_text:\n', response_text)
    #print(predicted_text)



    open_spans = get_open_span(action_text)
    print('\open_spans:\n', open_spans)

    # handling images
    if venuename:
        result['has_image'] = 'True'
        images = img_handler_obj.get_imgs_url(query=venuename + "in Singapore", num_of_img=5)
        result['image'] = images[0]
        print(images)

    delex_system = '<|system|> {}'.format(response_text)
    context = context + ' ' + delex_system

    turn += 1
    prev_beliefs = beliefs
    result['Generated_belief'] = ' '.join(belief_text)
    result['Generated_action'] = ' '.join(action_text)
    result['response']  = response_text[10:]
    session['result'] = result
    return result



@app.route('/')
def index():
    session['state'] = 'start'
    session['label'] = ''
    session['result'] = {}
    return render_template('index2.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global survey_result
    message = request.form['message']
    response_text = Create_message(message)
    #print('\nRESPONSE TEXT ', response_text)
    if end_of_chat ==0:
        return jsonify(response_text)
    if end_of_chat ==1:
        question1 = ' '.join(("Please take a minute and fill out the following survey.<br>",
                          "Question 1 about Fluency:<br>",
                          "5: No repetitions at all.<br>",
                          "4: Only one repetition appeared in the response.<br>",
                          "3: Two repetitions appeared in the response.<br>",
                          "2: Three repetitions appeared in the response.<br>",
                          "1: More than three repetitions appeared in the response.<br>"))
        r = {'response': question1, 'status': 'off', 'has_image': 'False'}
        result['response']  = question1
        response_text = r
        session['result'] = r
        return jsonify(response_text)
    if end_of_chat ==2:
        survey_result += message
        question2 = ' '.join((
                          "Question 2 about Correctness of text response:<br>",
                          "5: Correct dialogue flow, correct grammar, and provide correct entities.<br>",
                          "4: Correct dialogue flow, grammar but small mistakes in provided entities.<br>",
                          "3: Noticeable mistakes in dialogue flow or grammar or provided entities but acceptable.<br>",
                          "2: Poor dialogue flow, grammar, and providing entities.<br>",
                          "1: Wrong dialogue flow, grammar, and wrong entities.<br>"))
        r = {'response': question2, 'status': 'off', 'has_image': 'False'}
        result['response']  = question2
        response_text = r
        session['result'] = r
        return jsonify(response_text)
    if end_of_chat ==3:
        survey_result += message
        question3 = ' '.join((
                          "Question 3 about Correctness of image response:<br>",
                          "5: 100% helpful in illustrating the recommendation and in your decision-making.<br>",
                          "4: 75% helpful in illustrating the recommendation and in your decision-making.<br>",
                          "3: 50% helpful in illustrating the recommendation and in your decision-making.<br>",
                          "2: 25% helpful in illustrating the recommendation and in your decision-making.<br>",
                          "1: 0% helpful in illustrating the recommendation and in your decision-making.<br>"))
        r = {'response': question3, 'status': 'off', 'has_image': 'False'}
        result['response']  = question3
        response_text = r
        session['result'] = r
        return jsonify(response_text)
    if end_of_chat ==4:
        survey_result += message
        question4 = ' '.join((
                          "Question 4 about 4 Humanlikeness:<br>",
                          "5: The response is 100% like generated by humans.<br>",
                          "4: The response is 75% like generated by humans.<br>",
                          "3: The response is 50% like generated by humans.<br>",
                          "2: The response is 25% like generated by humans.<br>",
                          "1: The response is 0% like generated by humans.<br>"))
        r = {'response': question4, 'status': 'off', 'has_image': 'False'}
        result['response']  = question4
        response_text = r
        session['result'] = r
        return jsonify(response_text)
    if end_of_chat ==5:
        survey_result += message
        logging.warning(f'received feedback in order of the question: {survey_result}')
        question5 = "We appreciate the time you took to share your feedback.<br>MMTOD CS team."
        r = {'response': question5, 'status': 'off', 'has_image': 'False'}
        result['response']  = question5
        response_text = r
        session['result'] = r
        return jsonify(response_text)
    return ""
