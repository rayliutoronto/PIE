import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

placeholder_name = 'a'
operation_name = 'add'

a = tf.placeholder(tf.int32, name=placeholder_name)
c = tf.placeholder(tf.int32, name='c')
b = tf.constant(20)

# This is our model
add = tf.add(tf.add(a, c, name='addddd'), b, name=operation_name)

with tf.Session() as sess:

    # Run a few operations to make sure our model works
    ten_plus_two = sess.run(add, feed_dict={a: 2, c: 5})
    print('20 + 2 + 5 = {}'.format(ten_plus_two))

    ten_plus_ten = sess.run(add, feed_dict={a: 10, c: 5})
    print('20 + 10 + 5 = {}'.format(ten_plus_ten))

    # Pick out the model input and output
    a_tensor = sess.graph.get_tensor_by_name(placeholder_name + ':0')
    sum_tensor = sess.graph.get_tensor_by_name(operation_name + ':0')

    model_input1 = build_tensor_info(a)
    model_input2 = build_tensor_info(c)
    model_output = build_tensor_info(add)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={'anystring': model_input1, 'test': model_input2},
        outputs={'str': model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder('./models/simple_model/4')

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()