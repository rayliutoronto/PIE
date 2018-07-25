import numpy as np
from tensorflow.python.training import session_run_hook


class EarlyStoppingHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step.
        check if r1.10 has similar function when that version is ready
    """

    def __init__(self, estimator, input_fn, min_delta=0, patience=0):
        self.estimator = estimator
        self.input_fn = input_fn
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.monitor_op = np.greater
        self.min_delta *= 1
        self.best = -np.Inf

    # def begin(self):
    #     # Convert names to tensors if given
    #     graph = tf.get_default_graph()
    #     self.monitor = graph.as_graph_element(self.monitor)
    #     if isinstance(self.monitor, tf.Operation):
    #         self.monitor = self.monitor.outputs[0]

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self.run_context = run_context
        # return session_run_hook.SessionRunArgs(self.monitor)

    # def after_run(self, run_context, run_values):
    #     current = run_values.results
    #
    #     if self.monitor_op(current - self.min_delta, self.best):
    #         self.best = current
    #         self.wait = 0
    #     else:
    #         self.wait += 1
    #         if self.wait >= self.patience:
    #             run_context.request_stop()

    def end(self, session):
        current = self.estimator.evaluate(input_fn=self.input_fn)['accuracy']
        print('accuracy: ', current)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.run_context.request_stop()
