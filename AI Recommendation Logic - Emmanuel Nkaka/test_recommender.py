import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recommender as r


class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.roles = r.load_roles()

    def test_cosine_identical_is_one(self):
        self.assertAlmostEqual(r.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_cosine_orthogonal_is_zero(self):
        self.assertEqual(r.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_zero_vector_safe(self):
        self.assertEqual(r.cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_idf_rare_beats_common(self):
        vocab = r.build_vocabulary(self.roles)
        idf = r.compute_idf(self.roles, vocab)
        self.assertGreater(idf["tensorflow"], idf["python"])

    def test_alias_mapping(self):
        self.assertEqual(r.normalize("Cloud Computing"), "cloud")
        self.assertEqual(r.normalize("  K8S "), "kubernetes")

    def test_devops_profile(self):
        res, _ = r.recommend(["docker", "kubernetes", "ci/cd"], self.roles)
        self.assertEqual(res[0][0], "DevOps Engineer")

    def test_top_n_and_sorted(self):
        res, _ = r.recommend(["python", "sql", "machine learning"], self.roles, 3)
        self.assertEqual(len(res), 3)
        scores = [s for _, s, _, _ in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= s <= 1.0 + 1e-9 for s in scores))

    def test_unknown_skills_flagged(self):
        _, unknown = r.recommend(["python", "docker", "cobol"], self.roles)
        self.assertEqual(unknown, ["cobol"])

    def test_cold_start_fallback(self):
        res, _ = r.recommend(["zzz", "yyy", "xxx"], self.roles)
        self.assertEqual(len(res), 3)


if __name__ == "__main__":
    unittest.main()
