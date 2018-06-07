import tensorflow as tf

a = tf.add(1,2,name="Adding_1_and_2")

b = tf.multiply(a, 3, name="Adding_a_and_3")
c = tf.add(b, a, name="Adding_b_5")
d = tf.multiply(c, 12, name = "Adding_c_and_12")


with tf.Session() as session:
    writer = tf.summary.FileWriter("output_prac", session.graph)
    print(session.run(d))
    writer.close()

    
    
