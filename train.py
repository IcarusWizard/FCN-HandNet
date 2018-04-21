from datatools import batch_data
from model import FCN_Handnet
from modeltools import findStep
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
win_unicode_console.enable()

if __name__ == '__main__':
    train_data_path = '.\\TrainData'
    validation_data_path = '.\\ValidationData'
    test_data_path = '.\\TestData'
    itertime = 10
            
    sess = tf.InteractiveSession()

    train_op, img, label, stage, loss, acc = FCN_Handnet()

    saver = tf.train.Saver()

    step = findStep('Model')
    
    if not step == 0:
        saver.restore(sess, './Model/model.ckpt-' + str(step))
    else:
        sess.run(tf.initialize_all_variables())

    for it in range(itertime):
        print('No.%d iteration' % (it + 1))

        for img_batch, label_batch in batch_data(train_data_path, 32):
            step += 1
            m = img_batch.shape[0]
            feed_dict = {img : img_batch, label : label_batch}
            if step % 100 == 0:
                _, loss_value, acc_value = sess.run([train_op, loss, acc], feed_dict=feed_dict) 
                print('In Train Step %d' % step)
                loss_value = [loss_value[i] / m for i in range(3)] 
                print('loss = ', loss_value)
                acc_value = [acc_value[i] / m for i in range(3)]
                print('acc = ', acc_value)
            else:
                sess.run(train_op, feed_dict=feed_dict)
            if step % 10000 == 0:
                saver.save(sess, "Model/model.ckpt", global_step=step)
    saver.save(sess, "Model/model.ckpt", global_step=step)

        