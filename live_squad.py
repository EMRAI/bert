import collections
import tensorflow as tf

import run_squad
import tokenization
import modeling

from collections import namedtuple

from live_estimator import LiveEstimator


"""
A LiveEstimator for SQuAD data (or similarly formatted).
Draws heavily from code in run_squad.py (and uses functions directly to that module where possible).
"""


RawResult = namedtuple('RawResult', ['unique_id', 'membership'])


class LiveSquad(LiveEstimator):

    def __init__(self, flags):
        tf.logging.set_verbosity(tf.logging.INFO)

        self.bert_config = modeling.BertConfig.from_json_file(flags.bert_config_file)

        run_squad.validate_flags_or_throw(self.bert_config)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=flags.vocab_file, do_lower_case=flags.do_lower_case)

        self.flags = flags
        super(LiveSquad, self).__init__()

    def my_get_estimator(self):
        tpu_cluster_resolver = None
        if self.flags.use_tpu and self.flags.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.flags.tpu_name, zone=self.flags.tpu_zone, project=self.flags.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.flags.master,
            model_dir=self.flags.output_dir,
            save_checkpoints_steps=self.flags.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.flags.iterations_per_loop,
                num_shards=self.flags.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        model_fn = run_squad.model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.flags.init_checkpoint,
            learning_rate=self.flags.learning_rate,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=self.flags.use_tpu,
            use_one_hot_embeddings=self.flags.use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
        return tf.contrib.tpu.TPUEstimator(
            use_tpu=self.flags.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.flags.train_batch_size,
            predict_batch_size=self.flags.predict_batch_size)

    def my_create_examples(self, data_object):
        """
        Modified version of read_squad_examples from run_squad.
        Note that this returns feature objects, not example objects. The feature TENSORS themselves are made elsewhere.
        :param data_object: equivalent object to the 'data' section of the SQuAD JSON scheme
        :return: a list of `SquadExample`s
        """
        def is_whitespace(c):
            return c in " \t\r\n" or ord(c) == 0x202F

        examples = []
        for entry in data_object:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    examples.append(run_squad.SquadExample(
                        qas_id=qa["id"],
                        question_text=qa["question"],
                        doc_tokens=doc_tokens,
                        orig_answer_text=None,
                        start_position=None,
                        end_position=None,
                        is_impossible=False)
                    )

        feature_objects = []
        run_squad.convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.flags.max_seq_length,
            doc_stride=self.flags.doc_stride,
            max_query_length=self.flags.max_query_length,
            is_training=False,
            output_fn=feature_objects.append)
        return feature_objects

    def my_create_features(self, example):
        """
        TODO: document
        Pieces are split up right now...might be useful for debugging (make examples, then make features)
        :param data_object:
        :return:
        """
        return {
                "unique_ids": example.unique_id,
                "input_mask": example.input_mask,
                "input_ids": example.input_ids,
                "segment_ids": example.segment_ids,
                }

    def my_result_to_output(self, feature, result):
        unique_id = int(result["unique_ids"])
        membership = [float(x) for x in result["membership"]]
        # start_logits = [float(x) for x in result["start_logits"].flat]
        # end_logits = [float(x) for x in result["end_logits"].flat]
        raw_result = RawResult(
                    unique_id=unique_id,
                    # start_logits=start_logits,
                    # end_logits=end_logits,
                    membership=membership)
        return get_predictions(feature, raw_result, self.flags.n_best_size, self.flags.max_answer_length)

    def my_data_types(self):
        return {'unique_ids': tf.int32, 'input_ids': tf.int32, 'segment_ids': tf.int32, 'input_mask': tf.int32}

    def my_output_shapes(self):
        return {'unique_ids': tf.TensorShape([]),
                'input_ids': tf.TensorShape([self.flags.max_seq_length]),
                'segment_ids': tf.TensorShape([self.flags.max_seq_length]),
                'input_mask': tf.TensorShape([self.flags.max_seq_length])}

    def get_result_id(self, result):
        return result['unique_ids']

    def set_feature_id(self, feature, unique_id):
        feature['unique_ids'] = unique_id


def get_nbest_bounds_from_membership(membership_logits, n_best_size=1):
    """
    Return possible inclusive start, exclusive end indices given a list of membership logits.
    :param membership_logits:
    :return: two lists, each of length n (in nbest)
    """
    # TODO: include heuristic for choosing bounds (not just min/max)
    # TODO: implement nbest in heuristic too
    indices = [i for i, m in enumerate(membership_logits) if m > 0]
    start_index = min(indices) if len(indices) else 0
    end_index = max(indices) if len(indices) else 0
    return [start_index], [end_index]


def get_predictions(example, result, n_best_size, max_answer_length):
    """
    This function has been mostly copied from run_squad.py.
    Unfortunate, but I needed to return local variables from that function.
    :param all_examples:
    :param all_features:
    :param all_results:
    :param n_best_size:
    :param max_answer_length:
    :param null_score_diff_threshold:
    :return:
    """

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score

    # start_indexes = run_squad._get_best_indexes(result.start_logits, n_best_size)
    # end_indexes = run_squad._get_best_indexes(result.end_logits, n_best_size)
    start_indexes, end_indexes = get_nbest_bounds_from_membership(result.membership, n_best_size)
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(example.tokens):
                continue
            if end_index >= len(example.tokens):
                continue
            if start_index not in example.token_to_orig_map:
                continue
            if end_index not in example.token_to_orig_map:
                continue
            if not example.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    # feature_index=feature_index,
                    feature_index=0,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=1,
                    end_logit=1))
                    # start_logit=result.start_logits[start_index],
                    # end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_offset", "end_offset"])

    seen_predictions = set()
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
          break
        # feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            start_offset = example.token_to_orig_map[pred.start_index]
            end_offset = example.token_to_orig_map[pred.end_index] + 1
            try:
                final_text = ' '.join(example.doc_tokens[start_offset:end_offset])
            except AttributeError:
                final_text = ''
            if (start_offset, end_offset) in seen_predictions:
                continue
            seen_predictions.add((start_offset, end_offset))
        else:
            final_text = ""
            start_offset = 0
            end_offset = 0
            seen_predictions.add((0, 0))

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
                start_offset=start_offset,
                end_offset=end_offset))

    # if we didn't include the empty option in the n-best, include it
    if (0, 0) not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit,
                start_offset=0,
                end_offset=0))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_offset=0, end_offset=0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = run_squad._compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["start_offset"] = entry.start_offset
        output["end_offset"] = entry.end_offset
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    assert len(nbest_json) >= 1

    return nbest_json
