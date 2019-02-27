import os
import sys

# TODO: OpenAPI + connexion, not plain flask
import flask

import run_squad
from live_squad import LiveSquad


app = flask.Flask(__name__)
client = None

BERT_QA_HOST = os.environ.get('BERT_QA_HOST', '0.0.0.0')
BERT_QA_PORT = os.environ.get('BERT_QA_PORT', '8002')


def bad_request(msg):
    response = flask.jsonify({'message': msg})
    response.status_code = 400
    return response


def bert_squad_out_to_emrai(squad_out):
    """
    Convert the SQuAD-type output from our BERT wrapper to the output type the EMR.AI Q&A system has historically used.
    :param squad_out:
    :return:
    """
    best = squad_out[0]
    start, end = best['start_offset'], best['end_offset']
    return [[start, end]] if start < end else []


@app.route("/", methods=["POST"])
def simple_test():
    json_input = flask.request.get_json()
    if 'text' in json_input:
        context = json_input['text']
    elif 'tokens' in json_input:
        context = ' '.join(json_input['text'])
    else:
        return bad_request("Need 'text' or 'tokens' key in input object (ignores 'tokens' if 'text' present)")

    questions = json_input['qs']
    qas = [{'id': i, 'question': question} for i, question in enumerate(questions)]
    input_obj = [{'paragraphs': [{'context': context, 'qas': qas}]}]
    prediction = client.predict(input_obj)

    return_obj = json_input
    return_obj['answers'] = [bert_squad_out_to_emrai(ans) for ans in prediction]

    return flask.jsonify(return_obj)

if __name__ == '__main__':
    client = LiveSquad(run_squad.FLAGS)
    try:
        app.run(BERT_QA_HOST, int(BERT_QA_PORT))
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
