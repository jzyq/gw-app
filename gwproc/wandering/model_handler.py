import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import time
import numpy as np
import yaml

from utils.commons import ImageHandler, dec_timer
from core.exceptions import error_handler

from collections import OrderedDict
import scipy.linalg
import scipy
import lap
import argparse

from utils.log_config import get_logger
logger = get_logger()

_cur_dir_=os.path.dirname(__file__)

"""
BASETRACK
"""
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


"""
KALMAN_FILTER
"""
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


"""
MATCHING_FUNCTIONS
"""
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
def ious(boxes1, boxes2):
    """
    计算两组边界框之间的IoU（交并比）。

    参数:
    - boxes1: 第一组边界框，形状为 (N, 4)，其中N是边界框的数量。
    - boxes2: 第二组边界框，形状为 (K, 4)，其中K是边界框的数量。

    返回值:
    - 一个形状为 (N, K) 的数组，表示boxes1中的每个边界框与boxes2中每个边界框之间的IoU。
    """
    # N = boxes1.shape[0]
    # K = boxes2.shape[0]
    N = len(boxes1)
    K = len(boxes2)
    overlaps = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            box1 = boxes1[n]
            box2 = boxes2[k]

            # 计算重叠的边界
            ixmin = max(box1[0], box2[0])
            iymin = max(box1[1], box2[1])
            ixmax = min(box1[2], box2[2])
            iymax = min(box1[3], box2[3])

            iw = max(ixmax - ixmin + 1, 0)
            ih = max(iymax - iymin + 1, 0)
            inters = iw * ih

            # 合并面积
            uni = ((box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) +
                   (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) -
                   inters)

            overlaps[n, k] = inters / uni

    return overlaps
def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

"""
BYTE_TRACKER
"""
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tlwh_yolo = None

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.tlwh_yolo = new_track.tlwh
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh_yolox(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.tlwh_yolo is None:
            return self._tlwh.copy()
        ret = self.tlwh_yolo.copy()
        return ret

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
class BYTETracker(object):
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_thresh = 0.3
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 3000
        self.mot20 = False
        #self.det_thresh = args.track_thresh
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, bboxes,scores):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        ids = []
        for itracked, idet in matches:
            ids.append(idet)
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            ids.append(idet)
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            ids.append(idet)
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks """
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
def process_output(self, output, conf, iou):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf, :]
    scores = scores[scores > conf]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object

    boxes = self.extract_boxes(predictions)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # indices = nms(boxes, scores, self.iou_threshold)

    boxes0 = boxes[class_ids == 0, :]
    scores0 = scores[class_ids == 0]
    class_ids0 = class_ids[class_ids == 0]
    indices0 = self.nms(boxes0, scores0, iou)

    box_out = boxes0[indices0]
    class_out = class_ids0[indices0]
    score_out = scores0[indices0]
    return box_out, score_out, class_out


class BehaviorDetectWanderingHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()

        self.model_name = 'wandering'
        self.new_shape = [640, 640]

        # 设置缺省徘徊控制区域为全图像, 配置文件和接口调用中若有明确定义将覆盖缺省定义
        self.areas=[{"area_id": 0, "points": self.__class__.points2box([0,0], [self.new_shape[1]-1, self.new_shape[0]-1])}]

        self.read_config()

        self.classes = ['wandering']
        self.num_classes = 1
        
        """
        self.areas = [
            {
                "area_id": 0,
                "points": [[0, 0], [999999, 0], [999999, 999999], [0, 999999]]
            }
        ]
        """

        self.platform = platform
        
        if self.platform == 'ONNX':
            import onnxruntime as ort

            sess = ort.InferenceSession(
                os.path.join(_cur_dir_, 'models/wandering.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            logger.info(f'Model {self.model_name} Loaded')

        elif self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device_id = device_id
            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            sess = AclLiteModel(os.path.join(_cur_dir_, 'models/wandering.om'))

            logger.info(f'Model {self.model_name} Loaded on {self.device}')
        else:
            # TO DO: should not be here, should report an error
            pass
        
        self.inference_sessions = {'wandering': sess}
       
    @classmethod
    def points2box(cls,p1,p2):
        _left   = min(p1[0], p2[0])
        _right  = max(p1[0], p2[0])
        _top    = min(p1[1], p2[1])
        _bottom = max(p1[1], p2[1])
        
        return [[_left,_top],[_right,_top],[_right,_bottom],[_left,_bottom]]
        
    def read_config(self):
        # 读取配置文件
        _config_file = os.path.join(_cur_dir_, 'config.yaml')
        with open(_config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 访问参数
        self.conf = config['detect']['conf_thres']
        self.iou = config['detect']['iou_thres']
        self.filter_size = config['detect']['filter_size']
        
        """
        self.do_dedup = (config['dedup']['enable']==1)
        self.dedup_template_thres = config['dedup']['template_thres']
        self.dedup_iou_thres = config['dedup']['iou_thres']
        self.dedup_freq = config['dedup']['time_freq']
        """
            
        if config['areas'] is not None and config['areas'].items() is not None and len(config['areas'].items()) > 0:
            _areas = []
            
            for _key, _value in config['areas'].items():
                if len(_value) == 2: # (left,top) - (right,bottom)
                    _areas.append({"area_id": int(_key), "points": self.__class__.points2box(_value[0], _value[1])})
                else:
                    _areas.append({"area_id": int(_key), "points": _value})
            
            if len(_areas) != 0:
                self.areas=_areas
            
    def release(self):
        if self.platform == 'ASCEND':
            for _sess in self.inference_sessions:
                del _sess
            del self.resource
            
            logger.info(f'Model {self.model_name} Relased on {self.device}')

        else:
            logger.info(f'Model {self.model_name} Relased')

    def run_inference(self, image_files, extra_args=None):
        _images_data = []
        _images_error = ""

        image_result = True
        for _image_file in image_files:
            try:
                from ..gwproc import GWProc,GWProc_Result
            except:
                from gwproc import GWProc,GWProc_Result

            _result, _data = GWProc.read_image(_image_file)
            
            if _result:
                _encoded_str = base64.urlsafe_b64encode(_data)
                _images_data.append(_encoded_str.decode('utf8'))
            else:
                image_result = False
                _images_error += _data

        if image_result == False:
            return GWProc.result_json(self.model_name, GWProc_Result.IMAGE_FAIL, desc=_images_error)

        _areas=[]
        if extra_args is not None:
            if extra_args.get('objectList') is not None:
                if extra_args['objectList'][0].get('pos') is not None:
                    _pos_items = extra_args['objectList'][0].get('pos')
                    
                    for _i,_pos in enumerate(_pos_items):
                        if _pos.get('areas') is not None:
                            _point_list = _pos.get('areas')
                            
                            _area_points=[]
                            for _p in _point_list:
                                _area_points.append([_p["x"], _p["y"]])
                    
                            if len(_area_points) == 2:
                                _area_points = self.__class__.points2box(_area_points[0], _area_points[1])

                        _areas.append({"area_id": (_i+1), "points": _area_points})
                            
        if len(_areas) == 0: # 没有extra_args将会采用初始化中的areas
            _areas=self.areas

        payload = {
        "task_tag": "behavior_detect",
        "image_type": "base64",
        "images": _images_data,
            "extra_args": [
                {
                    "model": self.model_name,
                    'param': {
                        "areas": _areas
                    }
                }
            ]
        }

        try:
            data = self.preprocess(payload)
            data = self.inference(data)
            data = self.postprocess(data)

            #当前仅支持单输入单输出, 仅取[0]号元素
            _defects = data['data'][0]['defect_data']
            if len(_defects) == 0:
                return GWProc.result_json(self.model_name, GWProc_Result.INFER_ZERO_DETECT)
            else:
                _max_conf = 0.0
                _pos = []
                for _def in _defects:
                    _aid = _def['extra_info']['area_id']
                    #_defect_area=None
                    for _area in _areas:
                        if _area['area_id']== _aid:
                            _defect_points=_area['points']
                            break
                    if _defect_points is None:
                        raise ValueError(f'找到返回区域代码{_aid}对应监控区域, 要求监控区域为{_areas} ')
                    else:
                        _defect_areas_points=[]
                        for _p in _defect_points:
                            _defect_areas_points.append({"x":_p[0],"y":_p[1]})
                        _defect_areas={"areas": _defect_areas_points}

                    if _def['confidence'] > _max_conf:
                        _max_conf=_def['confidence']
                        _desc = _def['defect_desc']
                        _pos=[_defect_areas]+_pos
                    else:
                        _pos=_pos+[_defect_areas]

                return GWProc.result_json(self.model_name, GWProc_Result.INFER_AND_DETECT, value="1", desc=_desc, conf=_max_conf/100.0, pos=_pos)
        except Exception as e:
            _data = f"{self.model_name}: An error occurred: {e}"
            logger.error(_data)
            return GWProc.result_json(self.model_name, GWProc_Result.INFER_FAIL, _data)
        # data = self.preprocess(payload)
        # data = self.inference(data)
        # data = self.postprocess(data)

        # return json.dumps(data, indent=4, ensure_ascii=False)

    def preprocess(self, data, **kwargs):
        return data

    @error_handler
    def inference(self, data, *args, **kwargs):
        """推理"""
        return_datas = []
        areas = []
        image_type = data.get("image_type")
        images = data.get("images")
        filter_size = data.get("filter_size")
        extra_args = data.get("extra_args")
        if filter_size is None:
            filter_size = self.filter_size

        sess = self.inference_sessions.get('wandering')
        
        if self.platform == 'ONNX':
            input_name = sess.get_inputs()[0].name
            label_name = [i.name for i in sess.get_outputs()]

        if extra_args:
            for model_param in extra_args:
                model = model_param.get("model")
                if model == self.model_name:
                    param = model_param.get('param')
                    confidence = param.get('conf')
                    iou_thre = param.get('iou')
                    areas = param.get('areas')
                    filter_size2 = param.get('filter_size')
                    do_dedup = param.get('do_dedup')
                    time_freq = param.get('time_freq')
                    if confidence is None:
                        confidence = self.conf
                    if iou_thre is None:
                        iou_thre = self.iou
                    if (filter_size2 is not None) and (filter_size != filter_size2):
                        filter_size = filter_size2
                    if areas is None:
                        areas = self.areas
        else:
            confidence = self.conf
            iou_thre = self.iou

            do_dedup = None
            time_freq = None
            areas = self.areas

            if filter_size is None:
                filter_size = self.filter_size
            #kpt_thres = self.kpt_thres
                                    
        if image_type == "base64":
            im = np.zeros((6, 3, 640, 640), dtype = np.float32)
            for i ,base64_str in enumerate(images):
                img0 = self.base64_to_cv2(base64_str)
                img = self.prepare_input(img0, swapRB=False)
                im[i] = img
            
            if self.platform == 'ONNX':
                output = sess.run(label_name, {input_name: im}, **kwargs)[0]
            elif self.platform == 'ASCEND':
                output = sess.execute([im])[0]

            for i, base64_str in enumerate(images):
                img0 = self.base64_to_cv2(base64_str)
                return_datas.append(["img" + str(i + 1), img0, np.expand_dims(output[i], axis=0)])

        else:
            result = {}
            result["code"] = 400
            result["message"] = f"'model': '{image_type}'"
            result["time"] = int(time.time() * 1000)
            result["data"] = []
            return result
        print("filter_size：",filter_size)
        return return_datas, (confidence, iou_thre, areas, filter_size, do_dedup, time_freq)

    @error_handler
    def postprocess(self, data, *args, **kwargs):
        """后处理"""

        if isinstance(data, dict) and data['code'] == 400:
            return data

        finish_datas = {"code": 200, "message": "", "time": 0, "data": []}

        if data:
            data, param = data
            confidence, iou_thre, areas, filter_size, do_dedup, do_freq = param
            defect_data = []
            for area in areas:
                area_id = area.get('area_id')
                points = area.get('points')
                person_count = dict()

                tracker = BYTETracker(frame_rate=30)
                results = []
                record = []

                for i, img_data in enumerate(data):
                    img_tag, img_raw, preds = img_data
                    box_out, score_out, class_out = self.filter_by_size(self.process_output(preds, confidence, iou_thre),
                                                      filter_size=filter_size)
                    boxes = np.array(box_out)
                    scores = np.array(score_out)
                    online_tlxys, online_ids, online_scores = [], [], []
                    if len(box_out) > 0:
                        online_targets = tracker.update(boxes, scores)
                        # online_tlxys, online_ids, online_scores = [], [], []
                        for i, t in enumerate(online_targets):
                            tid = t.track_id
                            tscore = t.score
                            tlwh = t.tlwh_yolox.tolist()
                            tlxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                            foot1 = [tlxy[0], tlxy[3]]
                            foot2 = [tlxy[2], tlxy[3]]
                            if self.is_in_poly(foot1, points) or self.is_in_poly(foot2, points):
                                if tid not in person_count:
                                    person_count[tid] = 1
                                elif tid in person_count:
                                    person_count[tid] += 1
                                online_tlxys.append(tlxy)
                                online_ids.append(tid)
                                online_scores.append(tscore)
                    if img_tag == 'img6':
                        record.append(online_tlxys)
                        record.append(online_ids)
                        record.append(online_scores)
                wandering_person = [key for key, value in person_count.items() if value >= 3]
                if wandering_person:
                    for inx in wandering_person:
                        if inx in record[1]:
                            index = record[1].index(inx)
                            conf = int(record[2][index]*100)
                            defect_data.append({"defect_name": "wandering",
                                                'defect_desc': "场站有人员徘徊，请及时警告!",
                                                "confidence": conf,
                                                "class": "wandering",
                                                "extra_info": {"area_id": area_id,
                                                               "id": inx},
                                                "x1": int(record[0][index][0]), "y1": int(record[0][index][1]),
                                                "x2": int(record[0][index][2]), "y2": int(record[0][index][3]),
                                                })
            finish_datas["data"].append({"image_tag": "img6", "defect_data": defect_data})


        finish_datas["time"] = int(round(time.time() * 1000))

        return finish_datas

json_str = '''{
    "requestHostIp": "10.11.120.39",
    "requestHostPort": "8766",
    "requestId": "1234abcd-1a2b-4444-3c4d-1a2b3c4d5e6f",
    "objectList": [
        {
            "objectId": "123-1",
            "typeList": [
                "wcaqm",
                "wcgz",
                "hxq_gjbs",
                "xy"
            ],
            "imageUrlList": [
                "wcaqm1.jpg"
            ],
            "imageNormalUrlPath": "",
            "pos": [{
                "areas": [
                    {"x": 100, "y": 100},
                    {"x": 600, "y": 600}
                ]
            }]
        }
    ]
}'''

# Note: ftp服务器在首次实例化GWProc时设置，无法在此单独进行测试

"""
if __name__ == '__main__':
    import onnxruntime as ort

    input_images = [os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-8d4376beeb5048feb0cfb7ed798b6d71.png'),
                    os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-27fb1e1bd3844b07a18f7d2aac8bbc71.png'),
                    os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-93bfd984746c49e0b02cf33b7575bcb3.png'),
                    os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-ef7e24fc0da949f4af6a471209904250.png'),
                    os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-f325e0299dbd4c77ae71f5b4ebf7ad38.png'),
                    os.path.join(_cur_dir_,'test_case/2024-10-25T17.30.16-[539,119,611,278]-faf3ca75c7014e55ac307d8eddcf8616.png')
                    ]

    obj = BehaviorDetectWanderingHandler(platform='ONNX')
    
    extra_args = json.loads(json_str)
    results = obj.run_inference(input_images, extra_args=extra_args)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
"""