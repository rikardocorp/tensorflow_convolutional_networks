
import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create the variables
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
# ___step 0:___ reshape it to 4D-tensors
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1))
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3))
# ____step 1:____ create the summaries
gs_summary = tf.summary.image('Grayscale', w_gs_reshaped)
c_summary = tf.summary.image('Color', w_c_reshaped, max_outputs=5)
# ____step 2:____ merge all summaries
merged = tf.summary.merge_all()
# create the op for initializing all variables
init = tf.global_variables_initializer()
# launch the graph in a session
with tf.Session() as sess:
    # ____step 3:____ creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # initialize all variables
    sess.run(init)
    # ____step 4:____ evaluate the merged op to get the summaries
    summary = sess.run(merged)
    # ____step 5:____ add summary to the writer (i.e. to the event file) to write on the disc
    writer.add_summary(summary)
    print('Done writing the summaries')