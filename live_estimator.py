import threading
import tensorflow as tf

from abc import ABCMeta, abstractmethod

from six.moves import queue


"""
The LiveEstimator class is used for running jobs live through a TensorFlow Estimator.
The Estimator API is designed around batch decoding, but here we use a generator with queueing
    to "trick" it into operating in a service mode.

Usage:
    - Extend the LiveEstimator class; override all abstract methods to provide logic for building the model,
        creating the features, interpreting the output, and assigning unique IDs (for concurrency and queueing)
    - Have the server create an instance of your LiveEstimator implementation; call its `predict` method on each input.
"""


class LiveEstimator:

    __metaclass__ = ABCMeta

    ID_MAX = 2**31-1

    def __init__(self):
        self.final_results_dict = {}
        self.example_dict = {}
        self.jobs_queue = queue.Queue()
        self.events_dict = {}
        self.idcounter = 1

        # An event to be sure that we don't increment/assign ID on two different threads at once
        self.id_increment_event = threading.Event()
        self.id_increment_event.set()

        self.estimator = self.my_get_estimator()

        prediction_thread = threading.Thread(target=self._prediction_loop)
        prediction_thread.daemon = True
        prediction_thread.start()

    @abstractmethod
    def my_result_to_output(self, example, result):
        """
        Convert a single model result to the desired output format
        :param example: the example used to generate this result (if needed)
        :param result: a result as returned by the estimator
        :return: a single output object
        """

    @abstractmethod
    def my_create_examples(self, inputs):
        """
        Given inputs of a certain form, create a training example from it.
        Example can be in any format; it is an intermediate step in generating tensor features,
            but it is also available when creating the final output.
        :param inputs:
        :return:
        """

    @abstractmethod
    def my_create_features(self, example):
        """
        Given a single example, build features for it.
        :param inputs: iterable of inputs
        :return: a feature bundle for each input
        """

    @abstractmethod
    def my_get_estimator(self):
        """
        Initialize the estimator to use for prediction.
        :return: a tensorflow Estimator
        """

    @abstractmethod
    def my_data_types(self):
        """
        Get the data types for the input tensor.
        :return: probably a dictionary of data types
        """

    @abstractmethod
    def get_result_id(self, result):
        """
        Get a unique ID from the result. Model should save this somewhere in the result.
        :return: an int32 ID
        """

    @abstractmethod
    def set_feature_id(self, feature, unique_id):
        """
        Set the unique ID on the feature (for processing by the model).
        LiveEstimator uses its own internal unique ID scheme, but the model must persist it in the output.
        :return: an int32 ID
        """

    def my_output_shapes(self):
        """
        Provides the output shapes in the call to `Dataset.from_generator`.
        Override if necessary.
        :return:
        """
        return None

    def inputs_generator(self):
        while True:
            job = self.jobs_queue.get()
            print('JOB', job)
            yield job

    def create_input_fn(self):
        def input_fn(params):
            data_types = self.my_data_types()
            output_shapes = self.my_output_shapes()
            dataset = tf.data.Dataset.from_generator(self.inputs_generator, data_types, output_shapes)
            dataset = dataset.batch(1)
            return dataset
        return input_fn

    def _prediction_loop(self):
        for result in self.estimator.predict(self.create_input_fn(), yield_single_examples=False):
            unique_id = self.get_result_id(result)[0]   # single batch, hence [0]
            feature = self.example_dict[unique_id]
            output = self.my_result_to_output(feature, result)
            self.final_results_dict[unique_id] = output
            self.events_dict[unique_id].set()

    def _get_and_increment_id(self):
        """
        Increment the internal ID counter and return the (prior) value.
        Make this operation effectively atomic by wrapping in a threading Event.
        :return:
        """
        self.id_increment_event.clear()
        this_id = self.idcounter
        self.idcounter += 1
        if self.idcounter > self.ID_MAX:
            self.idcounter = 1
        self.id_increment_event.set()
        return this_id

    def predict(self, inputs):
        examples = self.my_create_examples(inputs)
        outputs = []
        for example in examples:
            feature = self.my_create_features(example)
            this_id = self._get_and_increment_id()
            this_job_event = threading.Event()
            self.events_dict[this_id] = this_job_event
            self.set_feature_id(feature, this_id)
            self.example_dict[this_id] = example
            self.jobs_queue.put(feature)
            # Generator will now fetch the job.
            # When done, it will set() the event in the events dict and save the result in the results dict.
            this_job_event.wait()
            result = self.final_results_dict[this_id]
            del self.final_results_dict[this_id]
            del self.example_dict[this_id]
            del self.events_dict[this_id]
            outputs.append(result)
        return outputs
