import unittest

from ecg_reasoning_benchmark.evaluators.heuristic import HeuristicEvaluator


class CurrentSchemaTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = HeuristicEvaluator(HeuristicEvaluator.parse_arguments([]))

    def test_renamed_binary_stage_accepts_correct_and_rejects_wrong(self):
        self.assertTrue(self.evaluator.validate("Yes", "Yes", "finding_identification"))
        self.assertFalse(self.evaluator.validate("Yes", "No", "finding_identification"))

    def test_renamed_decision_keeps_third_option(self):
        gt = "Further findings are required to confirm the diagnosis"
        self.assertTrue(self.evaluator.validate(gt, gt, "diagnostic_decision"))
        self.assertFalse(self.evaluator.validate(gt, "Yes", "diagnostic_decision"))

    def test_single_answer_grounding_list_is_preserved(self):
        for kind, answer in (("wave_grounding", "[1.48s - 1.69s]"),
                             ("measurement_grounding", "[210ms - 220ms]")):
            self.assertTrue(self.evaluator.validate([answer], answer, kind))
            self.assertFalse(self.evaluator.validate([answer], "unrelated answer", kind))

    def test_lead_grounding_preserves_multiple_answers(self):
        gt = ["Lead V1", "Lead V2"]
        self.assertTrue(self.evaluator.validate(gt, "V1, V2", "lead_grounding"))
        self.assertFalse(self.evaluator.validate(gt, "V1", "lead_grounding"))


if __name__ == "__main__":
    unittest.main()
