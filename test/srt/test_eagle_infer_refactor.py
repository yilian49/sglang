import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEagleBS1(CustomTestCase):
    num_questions = 10000

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-2-7b-chat-hf"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model",
                "lmzheng/sglang-EAGLE-llama2-chat-7B",
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "6",
                "--max-running-requests",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleBS1 -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.23
        )  # 0.3333 for 60 questions; 0.234 for 1319 questions


class TestEagleLargeBS(CustomTestCase):
    # num_questions = 10000
    num_questions = 32
    max_running_requests = 64
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        # "triton",
        "fa3",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model",
        "lmzheng/sglang-EAGLE-llama2-chat-7B",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-bs",
        *[str(i) for i in range(1, max_running_requests + 1)],
    ]

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-2-7b-chat-hf"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleLargeBS -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.23
        )  # 0.3333 for 60 questions; 0.234 for 1319 questions



class TestEagleLargeBSNoSD(TestEagleLargeBS):
    num_questions = 10000
    max_running_requests = 64
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-bs",
        *[str(i) for i in range(1, max_running_requests + 1)],
        "--disable-overlap-schedule",
    ]


class TestEagleLargeBSOverlapNoSD(TestEagleLargeBS):
    """
    Overlap scheduling ENABLED, speculative decoding DISABLED.
    """

    num_questions = 10000
    max_running_requests = 64
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-bs",
        *[str(i) for i in range(1, max_running_requests + 1)],
    ]



# E2E smoke for EAGLE3 on gpt-oss-20b with triton backend.
class TestEagleE3GptOss20B(CustomTestCase):
    """E2E smoke for EAGLE3 on gpt-oss-20b with triton backend.

    Mirrors the pattern in this file: launch a server with the provided
    flags and run the 5-shot GSM8K harness with a sanity accuracy bar.
    """

    num_questions = 10000

    @classmethod
    def setUpClass(cls):
        cls.model = "openai/gpt-oss-20b"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend", "triton",
                "--dtype", "bfloat16",
                "--mem-fraction-static", "0.85",
                "--tp", "1",
                "--speculative-algorithm", "EAGLE3",
                "--speculative-draft-model-path",
                "/data/shenggui/projects/SpecForge/outputs/perfect-blend-gptoss-20b-eagle3/epoch_1",
                "--speculative-num-steps", "3",
                "--speculative-eagle-topk", "2",
                "--speculative-num-draft-tokens", "6",
                "--cuda-graph-max-bs", "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleE3GptOss20B -- {metrics=}")
        self.assertGreater(metrics["accuracy"], 0.23)


if __name__ == "__main__":
    unittest.main()
