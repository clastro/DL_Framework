# 자동으로 default session을 지정
sess = tf.InteractiveSession() # 자동으로 default session을 지정 
sess.run(tf.global_variables_initializer()) 
print(c.eval())
print(sess.run(c)) 
sess.close() #close를 해 줘야함

# 세션을 열 경우 with로 열어줘야 함
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer()) 
  print(sess.run(c)) 
  print(c.eval())

