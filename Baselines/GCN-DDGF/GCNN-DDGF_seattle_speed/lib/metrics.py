import numpy as np
import tensorflow as tf


def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_rmse_np(preds, labels, gdt, null_val=0):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, gdt=gdt, null_val=null_val))


def masked_mse_np(preds, labels, gdt, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask1 = ~np.isnan(gdt)
            #mask2 = np.isnan(labels)
        else:
            mask1 = np.not_equal(gdt, null_val)
            #mask2 = np.equal(labels, null_val)
        #pos = np.where((mask1 == True) & (mask2 == True))
        pos = np.where(mask1 == True)
        mse = np.mean(np.square(gdt[pos] - preds[pos]))
        return mse


def masked_mae_np(preds, labels, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, gdt, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask1 = ~np.isnan(gdt)
            #mask2 = np.isnan(labels)
        else:
            mask1 = np.not_equal(gdt, null_val)
            #mask2 = np.equal(labels, null_val)
        #pos = np.where((mask1 == True) & (mask2 == True))
        pos = np.where(mask1 == True)
        mape = np.mean(np.abs((gdt[pos] - preds[pos]) / gdt[pos])) * 100
        '''
        mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
        '''
        return mape

# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse