import tensorflow as tf
from CNN_test import mfcc_train, mfcc_eval, mfcc_test, label_train, label_eval, label_test

BATCH_SIZE = 16 #샘플데이터 중 한번에 넘겨주는 데이터의 수
NUM_EPOCHS = 10 #전체 데이터를 돌며 학습하는 수
EMB_SIZE = 160

def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.Modekeys.PREDICT

    return 1

def mapping_fn(X, Y):
    data, label = {'x' : X}, Y
    return data, label

def train_fn():  ## Estimator 모델에서 사용되는 데이터 입력 파이프라인
    # Dataset생성 : 입력된 텐서로부터 slices 생성
    dataset = tf.data.Dataset.from_tensor_slices((mfcc_train, label_train))
    # 데이터들을 램던으로 섞음(큰수 입력)
    dataset = dataset.shuffle(buffer_size=len(mfcc_train))
    # 데이터를 읽어올 개수를 지정함
    dataset = dataset.batch(BATCH_SIZE)
    # 데이터 적절한 형식으로 매핑
    dataset = dataset.map(mapping_fn)
    # 데이터를 읽다가 마지막에 도달하면 다시 처음부터 조회
    datset = dataset.repeat(count=NUM_EPOCHS)
    # iterator 생성
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()  # 다음 항목에 연결되어 있는 tf.Tensor 객체를 리턴

def eval_fn():
    dataset = tf.data.Dataset.from_tensor_slices((mfcc_eval, label_eval))
    dataset = dataset.shuffle(buffer_size=len(mfcc_eval))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def test_fn():
    dataset = tf.data.Dataset.from_tensor_slices((mfcc_test, label_test))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

