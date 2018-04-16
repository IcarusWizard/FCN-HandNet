from datatools import batch_data
from model import FCN_Handnet
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console
win_unicode_console.enable()

if __name__ == '__main__':
    train_data_path = '.\\TestData'
    validation_data_path = '.\\ValidationData'
    test_data_path = '.\\ValidationData'
    itertime = 10
            
    sess = tf.InteractiveSession()

    train_op, img, label, stage1_loss, stage2_loss, stage3_loss, stage1_acc, stage2_acc, stage3_acc = FCN_Handnet(96, 3)
    loss = [stage1_loss, stage2_loss, stage3_loss]
    acc = [stage1_acc, stage2_acc, stage3_acc]

    sess.run(tf.initialize_all_variables())

    step = 0
    for it in range(itertime):
        print('No.%d iteration' % (it + 1))

        for img_batch, label_batch in batch_data(train_data_path, 16):
            step += 1
            m = img_batch.shape[0]
            feed_dict = {img : img_batch, label : label_batch}
            if step % 10 == 0:
                _, loss_value, acc_value = sess.run([train_op, loss, acc], feed_dict=feed_dict) 
                print('In Train Step %d' % step)
                loss_value = [loss_value[i] / m for i in range(3)] 
                print('loss = ', loss_value)
                acc_value = [acc_value[i] / m for i in range(3)]
                print('acc = ', acc_value)
            else:
                sess.run(train_op, feed_dict=feed_dict)

        