import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/camilo/Programacion/master_thesis')

from keras.models import Model
from keras.layers import Conv3D,PReLU, Conv3DTranspose,add,concatenate,Input,Dropout,BatchNormalization
import tensorflow as tf
from model.config import *

def resBlock(conv,stage,keep_prob,stage_num=5):#收缩路径
    
    inputs=conv
    
    for _ in range(3 if stage>3 else stage):
        conv=PReLU()(BatchNormalization()(Conv3D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
        #print('conv_down_stage_%d:' %stage,conv.get_shape().as_list())#输出收缩路径中每个stage内的卷积
    conv_add=PReLU()(add([inputs,conv]))
    #print('conv_add:',conv_add.get_shape().as_list())
    conv_drop=Dropout(keep_prob)(conv_add)
    
    if stage<stage_num:
        conv_downsample=PReLU()(BatchNormalization()(Conv3D(16*(2**stage), 2, strides=(2, 2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop)))
        return conv_downsample,conv_add#返回每个stage下采样后的结果,以及在相加之前的结果
    else:
        return conv_add,conv_add#返回相加之后的结果，为了和上面输出保持一致，所以重复输出
        
def up_resBlock(forward_conv,input_conv,stage):#扩展路径
    
    conv=concatenate([forward_conv,input_conv],axis = -1)
    print('conv_concatenate:',conv.get_shape().as_list())
    for _ in range(3 if stage>3 else stage):
        conv=PReLU()(BatchNormalization()(Conv3D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
        print('conv_up_stage_%d:' %stage,conv.get_shape().as_list())#输出扩展路径中每个stage内的卷积
    conv_add=PReLU()(add([input_conv,conv]))
    if stage>1:
        conv_upsample=PReLU()(BatchNormalization()(Conv3DTranspose(16*(2**(stage-2)),2,strides=(2, 2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(conv_add)))
        return conv_upsample
    else:
        return conv_add

def build_vnet(pretrained_weights = None,input_size = (256,256,1),num_class=1,is_training=True,stage_num=5,thresh=0.5):#二分类时num_classes设置成1，不是2，stage_num可自行改变，也即可自行改变网络深度
    keep_prob = 1.0 if is_training else 1.0#不使用dropout
    features=[]
    input_model = Input(input_size)
    x=PReLU()(BatchNormalization()(Conv3D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input_model)))
    
    for s in range(1,stage_num+1):
        x,feature=resBlock(x,s,keep_prob,stage_num)#调用收缩路径
        features.append(feature)
        
    conv_up=PReLU()(BatchNormalization()(Conv3DTranspose(16*(2**(s-2)),2,strides=(2, 2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(x)))
    
    for d in range(stage_num-1,0,-1):
        conv_up=up_resBlock(features[d-1],conv_up,d)#调用扩展路径
    if num_class>1:
        conv_out=Conv3D(num_class, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
    else:
        conv_out=Conv3D(num_class, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
    
    model=Model(inputs=input_model,outputs=conv_out)
    
    #plot_model(model, to_file='model.png')
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

def main():
    config = get_config_local_path()#get_config_test()
    model=build_vnet(input_size = config.image_size,num_class=config.n_classes,is_training=True,stage_num=2)
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file='vnet_model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

if __name__ == "__main__":
    main()