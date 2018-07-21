import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

placeholder_name = 'a'
operation_name = 'add'

a = tf.placeholder(tf.int32, name=placeholder_name)
b = tf.constant(20)

# This is our model
add = tf.add(a, b, name=operation_name)

with tf.Session() as sess:

    # Run a few operations to make sure our model works
    ten_plus_two = sess.run(add, feed_dict={a: 2})
    print('10 + 2 = {}'.format(ten_plus_two))

    ten_plus_ten = sess.run(add, feed_dict={a: 10})
    print('10 + 10 = {}'.format(ten_plus_ten))

    # Pick out the model input and output
    a_tensor = sess.graph.get_tensor_by_name(placeholder_name + ':0')
    sum_tensor = sess.graph.get_tensor_by_name(operation_name + ':0')

    model_input = build_tensor_info(a)
    model_output = build_tensor_info(add)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={'anystring': model_input},
        outputs={'str': model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder('./models/simple_model/3')

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()