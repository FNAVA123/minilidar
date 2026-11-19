"""Geometric regression checks; no model download or video required."""
import unittest
import numpy as np
import cv2
from mini_lidar_orb_depth import backproject, estimate_motion, world_from_current, camera_parameters, descriptor_matches, ReferenceTracker


class MiniLidarGeometryTest(unittest.TestCase):
    def tracker_fixture(self, matcher=None):
        from argparse import Namespace
        args = Namespace(min_reference_features=200, min_reference_inliers=30,
                         reference_interval=.5, reference_count=2, reference_max_age=8.,
                         stride=5, max_step=5., max_rotation=30., min_inliers=20,
                         recovery_min_inliers=40, min_depth=.25, max_depth=15., features='orb')
        pixels = [(x, y) for x in np.linspace(10, 630, 20) for y in np.linspace(10, 470, 10)]
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pixels]
        descriptors = np.zeros((200, 32), np.uint8)
        kwargs = {} if matcher is None else {'matcher': matcher}
        tracker = ReferenceTracker(np.eye(3), (640, 480), 10., args, **kwargs)
        return tracker, keypoints, descriptors

    def test_weak_frame_keeps_original_chain_but_does_not_replace_backup(self):
        tracker, keys, descriptors = self.tracker_fixture()
        depth = np.ones((480, 640), np.float32)
        tracker.accept(0, keys, descriptors, depth, np.eye(4), {'reason': 'initial'})
        pose = np.eye(4)
        pose[0, 3] = 1
        promoted = tracker.accept(5, keys[:30], descriptors[:30], depth, pose,
                                  {'reason': 'accepted', 'inliers': 25})
        self.assertFalse(promoted)
        self.assertEqual(tracker.primary[0], 5)
        self.assertEqual([r[0] for r in tracker.bank], [0])
        np.testing.assert_allclose(tracker.last_pose, pose)
        missing, _ = tracker.locate([], None, 6)
        self.assertIsNone(missing)
        # A failed attempt cannot update the stored pose or reference.
        self.assertEqual(tracker.last_frame, 5)
        self.assertEqual(tracker.primary[0], 5)

    def test_recovery_from_backup_uses_its_world_pose(self):
        def matcher(reference, keys, descriptors, K, args):
            if reference[3][0, 3] == 1:
                return None, {'reason': 'few_matches', 'matches': 2}
            transform = np.eye(4)
            transform[0, 3] = -2
            return transform, {'reason': 'accepted', 'inliers': 60, 'matches': 80}
        tracker, keys, descriptors = self.tracker_fixture(matcher)
        depth = np.ones((480, 640), np.float32)
        tracker.accept(0, keys, descriptors, depth, np.eye(4), {'reason': 'initial'})
        pose = np.eye(4)
        pose[0, 3] = 1
        tracker.accept(5, keys, descriptors, depth, pose, {'reason': 'accepted', 'inliers': 100})
        recovered, stats = tracker.locate(keys, descriptors, 10, recovering=True)
        self.assertTrue(stats['recovered'])
        self.assertEqual(stats['reference_frame'], 0)
        self.assertEqual(stats['reference_attempts'], 2)
        self.assertAlmostEqual(recovered[0, 3], 2)

    def test_reference_bank_is_bounded_and_weak_frames_are_excluded(self):
        tracker, keys, descriptors = self.tracker_fixture()
        pose, stats = tracker.locate(keys[:10], descriptors[:10], 0)
        np.testing.assert_allclose(pose, np.eye(4))
        self.assertEqual(stats['reason'], 'initial')
        depth = np.ones((480, 640), np.float32)
        self.assertFalse(tracker.accept(0, keys[:10], descriptors[:10], depth, pose, stats))
        self.assertEqual(tracker.bank, [])
        for frame in (0, 5, 10, 15):
            tracker.accept(frame, keys, descriptors, depth, np.eye(4), {'reason': 'initial'})
        self.assertEqual([r[0] for r in tracker.bank], [10, 15])

    def test_recovery_rejects_pose_jump_without_mutating_reference(self):
        def matcher(reference, keys, descriptors, K, args):
            if reference[3][0, 3] == 1:
                return None, {'reason': 'few_matches', 'matches': 2}
            transform = np.eye(4)
            transform[0, 3] = -40
            return transform, {'reason': 'accepted', 'inliers': 100, 'matches': 120}
        tracker, keys, descriptors = self.tracker_fixture(matcher)
        depth = np.ones((480, 640), np.float32)
        tracker.accept(0, keys, descriptors, depth, np.eye(4), {'reason': 'initial'})
        previous = np.eye(4)
        previous[0, 3] = 1
        tracker.accept(5, keys, descriptors, depth, previous, {'reason': 'accepted', 'inliers': 100})
        pose, stats = tracker.locate(keys, descriptors, 10, recovering=True)
        self.assertIsNone(pose)
        self.assertEqual(stats['reason'], 'recovery_pose_jump')
        np.testing.assert_allclose(tracker.last_pose, previous)
        self.assertEqual(tracker.last_frame, 5)

    def test_successful_primary_match_never_consults_backups_or_changes_thresholds(self):
        calls = []
        def matcher(reference, keys, descriptors, K, args):
            calls.append((reference[3][0, 3], args.min_inliers, args.max_step))
            return np.eye(4), {'reason': 'accepted', 'inliers': 25, 'matches': 30}
        tracker, keys, descriptors = self.tracker_fixture(matcher)
        depth = np.ones((480, 640), np.float32)
        tracker.accept(0, keys, descriptors, depth, np.eye(4), {'reason': 'initial'})
        previous = np.eye(4)
        previous[0, 3] = 1
        tracker.accept(5, keys[:30], descriptors[:30], depth, previous,
                       {'reason': 'accepted', 'inliers': 25})
        pose, stats = tracker.locate(keys, descriptors, 10, recovering=True)
        self.assertEqual(calls, [(1., 20, 5.)])
        self.assertFalse(stats['backup_used'])
        np.testing.assert_allclose(pose, previous)

    def test_xfeat_matching_uses_float_similarity_and_mutual_pairs(self):
        descriptors = np.eye(4, 64, dtype=np.float32)
        current = descriptors[[2, 0, 3, 1]].copy()
        matches = descriptor_matches(descriptors, current, 'xfeat')
        self.assertEqual({(m.queryIdx, m.trainIdx) for m in matches}, {(0, 1), (1, 3), (2, 0), (3, 2)})
        self.assertEqual(descriptor_matches(descriptors, -descriptors, 'xfeat'), [])

    def test_pnp_outliers_and_camera_pose_direction(self):
        rng = np.random.default_rng(14)
        points = rng.uniform([-2, -1, 4], [2, 1, 12], (200, 3))
        K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
        rvec = np.array([.01, .04, -.02])
        tvec = np.array([.12, .02, -.25])
        pixels = cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2)
        pixels[:50] = rng.uniform([0, 0], [640, 480], (50, 2))
        cv2.setRNGSeed(0)
        motion, stats = estimate_motion(points, pixels, K)
        self.assertIsNotNone(motion, stats)
        self.assertGreaterEqual(stats['inliers'], 145)
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(rvec)[0]
        expected[:3, 3] = tvec
        np.testing.assert_allclose(motion, expected, atol=1e-5)
        previous = np.eye(4)
        previous[:3, 3] = [1, 2, 3]
        pose = world_from_current(previous, motion)
        # A stationary landmark must agree through either coordinate chain.
        landmark = np.r_[points[100], 1]
        np.testing.assert_allclose(pose @ motion @ landmark, previous @ landmark, atol=1e-7)

    def test_backprojection_and_resized_calibration(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'camera.yaml'
            source.write_text('Camera.width: 1226\nCamera.height: 370\nCamera.fx: 700\nCamera.fy: 700\nCamera.cx: 600\nCamera.cy: 180\n')
            K, _ = camera_parameters(source, (1226, 370), (613, 185))
            np.testing.assert_allclose(K, [[350, 0, 300], [0, 350, 90], [0, 0, 1]])
            pixel = np.array([[K[0, 2], K[1, 2]], [K[0, 2] + K[0, 0], K[1, 2]]])
            np.testing.assert_allclose(backproject(pixel, np.array([2., 3.]), K), [[0, 0, 2], [3, 0, 3]])
            with self.assertRaises(ValueError):
                camera_parameters(source, (640, 480), (320, 240))

    def test_bad_tracking_is_rejected(self):
        motion, stats = estimate_motion(np.zeros((5, 3)), np.zeros((5, 2)), np.eye(3))
        self.assertIsNone(motion)
        self.assertEqual(stats['reason'], 'few_matches')


if __name__ == '__main__':
    unittest.main()
