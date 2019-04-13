""" Proximal Policy Optimization Agent """
import tensorflow as tf

from deeprl.agents import VPG
from deeprl.tf_utils import tfph


class PPO(VPG):
    """ PPO-clip implementation using early-stopping. Here early-stopping
    means that if the KL divergence of the new policy from the old reaches
    some threshold, policy training is halted """


    train_ph_keys = ['obs', 'act', 'logp', 'adv', 'ret']

    def __init__(self, pi_lr=3e-4, val_lr=1e-3, hidden_sizes=(64, 64),
                 activation=tf.tanh, val_train_iters=80, sess=None,
                 close_sess=True, clip_ratio=0.2, pi_train_iters=80,
                 target_kl=0.01):
        super().__init__(pi_lr=pi_lr, val_lr=val_lr,
                         hidden_sizes=hidden_sizes, activation=activation,
                         val_train_iters=val_train_iters, sess=sess,
                         close_sess=close_sess)
        self.clip_ratio, self.target_kl = clip_ratio, target_kl
        self.pi_train_iters = pi_train_iters
        self.kl_divergence = None

    def create_placeholders(self, obs_space, act_space):
        """ we need logp for the training the policy loss, so we'll add a
        placeholder for it here """
        super().create_placeholders(obs_space, act_space)
        self.placeholders['logp'] = tfph(None, name='logp')

    def build_policy_loss(self, logp, placeholders, learning_rate):
        """ overwrite parent method. Build the PPO policy loss """
        # prob of action taken according to the previous policy
        p_old = tf.exp(placeholders['logp'])
        # prob of action taken according to current policy
        p_new = tf.exp(logp)
        surrogate = placeholders['adv']*p_new/p_old
        max_surrogate = self.build_max_surrogate(
            self.clip_ratio, placeholders['adv'])
        clipped_surrogate = self.build_clipped_surrogate(
            surrogate, max_surrogate)
        pi_loss = -tf.reduce_mean(clipped_surrogate)
        pi_loss = -tf.reduce_mean(surrogate)
        pi_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(pi_loss)
        """ We additionally need to calculate the kl divergence of the new
        policy from the old for use as an early-stopping criterion during
        policy training """
        self.kl_divergence = p_old*(placeholders['logp'] - logp)
        return pi_loss, pi_train_op

    @staticmethod
    def build_max_surrogate(clip_ratio, adv):
        """ build the maximum surrogate
        max_surrogate = g(clip_ratio, adv)
        = (1 + sign(adv)*clip_ratio)*adv """
        return (1 + clip_ratio*tf.math.sign(adv))*adv

    @staticmethod
    def build_clipped_surrogate(surrogate, max_surrogate):
        """ build the clipped surrogate
        clipped_surrogate = minimum(surrogate, max_surrogate) """
        return tf.minimum(surrogate, max_surrogate)

    def update_policy(self, feed_dict):
        """ update the policy based on the replay-buffer data in feed_dict,
        and stop early if the kl divergence reaches the target """
        for _ in range(self.pi_train_iters):
            kl_div, _ = self.sess.run((self.kl_divergence, self.pi_train_op),
                                      feed_dict=feed_dict)
            # if we reach the target kl_div, halt training
            if kl_div >= self.target_kl:
                break
