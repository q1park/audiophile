import tensorflow as tf
import numpy as np

def _pairwise_distances(embeddings, squared=False):
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

def _anchorpositivemask(labels):
    labels = labels.numpy()
    maskAP = labels[:,None] == labels[None:]
    maskAP[np.eye(labels.size,dtype=bool)] = 0
    return tf.convert_to_tensor(maskAP, dtype=tf.float32)
    
def _anchornegativemask(labels):
    labels = labels.numpy()
    maskAN = labels[:,None] != labels[None:]
    return tf.convert_to_tensor(maskAN, dtype=tf.float32)

def _tripletmask(labels):
    labels = labels.numpy()
    maskAP = labels[:,None,None] == labels[None,:,None]
    maskAN = labels[:,None,None] != labels[None,None,:]
    maskAPN = np.logical_and(maskAP, maskAN)
    
    maskAPN.transpose(0,2,1)[np.eye(labels.size,dtype=bool), :] = 0
    maskAPN[np.eye(labels.size,dtype=bool), :] = 0
    maskAPN[:, np.eye(labels.size,dtype=bool)] = 0
    return tf.convert_to_tensor(maskAPN, dtype=tf.float32)

def batch_all(embeddings, labels, alpha = 0.2, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + alpha

    mask = _tripletmask(labels)

    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss

def batch_hard(embeddings, labels, alpha = 0.2, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _anchorpositivemask(labels)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    mask_anchor_negative = _anchornegativemask(labels)
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + alpha, 0.0)
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss