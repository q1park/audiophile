import tensorflow as tf
import numpy as np

def _pairwise_dist(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances

def _anchorpos_mask(labels):
    labels = labels.numpy()
    mask_anchorpos = labels[:,None] == labels[None:]
    mask_anchorpos[np.eye(labels.size,dtype=bool)] = 0
    return tf.convert_to_tensor(mask_anchorpos, dtype=tf.float32)
    
def _anchorneg_mask(labels):
    labels = labels.numpy()
    mask_anchorneg = labels[:,None] != labels[None:]
    return tf.convert_to_tensor(mask_anchorneg, dtype=tf.float32)

def _tripletmask(labels):
    labels = labels.numpy()
    mask_anchorpos = labels[:,None,None] == labels[None,:,None]
    mask_anchorneg = labels[:,None,None] != labels[None,None,:]
    mask_anchorposneg = np.logical_and(mask_anchorpos, mask_anchorneg)
    
    mask_anchorposneg.transpose(0,2,1)[np.eye(labels.size,dtype=bool), :] = 0
    mask_anchorposneg[np.eye(labels.size,dtype=bool), :] = 0
    mask_anchorposneg[:, np.eye(labels.size,dtype=bool)] = 0
    return tf.convert_to_tensor(mask_anchorposneg, dtype=tf.float32)

def batch_all(embeddings, labels, alpha = 0.2, squared=False):
    pairwise_dist = _pairwise_dist(embeddings, squared=squared)

    anchorpos_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchorpos_dist.shape[2] == 1, "{}".format(anchorpos_dist.shape)

    anchorneg_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchorneg_dist.shape[1] == 1, "{}".format(anchorneg_dist.shape)

    tripletloss = anchorpos_dist - anchorneg_dist + alpha

    mask = _tripletmask(labels)

    tripletloss = tf.multiply(mask, tripletloss)
    tripletloss = tf.maximum(tripletloss, 0.0)

    valid_triplets = tf.cast(tf.greater(tripletloss, 1e-16), tf.float32)
    npos_triplets = tf.reduce_sum(valid_triplets)
    tripletloss = tf.reduce_sum(tripletloss) / (npos_triplets + 1e-16)

    return tripletloss

def batch_hard(embeddings, labels, alpha = 0.2, squared=False):
    pairwise_dist = _pairwise_dist(embeddings, squared=squared)

    mask_anchorpos = _anchorpos_mask(labels)
    anchorpos_dist = tf.multiply(mask_anchorpos, pairwise_dist)
    hardest_pos_dist = tf.reduce_max(anchorpos_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_pos_dist", tf.reduce_mean(hardest_pos_dist))

    mask_anchorneg = _anchorneg_mask(labels)
    max_anchorneg_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchorneg_dist = pairwise_dist + max_anchorneg_dist * (1.0 - mask_anchorneg)
    hardest_neg_dist = tf.reduce_min(anchorneg_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_neg_dist", tf.reduce_mean(hardest_neg_dist))

    tripletloss = tf.maximum(hardest_pos_dist - hardest_neg_dist + alpha, 0.0)
    tripletloss = tf.reduce_mean(tripletloss)

    return tripletloss