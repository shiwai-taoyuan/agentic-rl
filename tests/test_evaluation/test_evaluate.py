from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.evaluation.evaluate import _extract_prompt


class TestExtractPrompt:
    def test_empty_conversation(self):
        assert _extract_prompt([]) == []

    def test_system_and_user_only(self):
        conversation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        assert _extract_prompt(conversation) == conversation

    def test_stops_at_assistant_tool_calls(self):
        conversation = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Query."},
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [{"id": "call1"}],
            },
            {"role": "tool", "content": "Result"},
        ]
        result = _extract_prompt(conversation)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_assistant_without_tool_calls_skipped(self):
        """Assistant messages without tool_calls are skipped by _extract_prompt."""
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _extract_prompt(conversation)
        # Only user messages are collected; assistant without tool_calls is skipped
        assert len(result) == 2
        assert result[0]["content"] == "Hi"
        assert result[1]["content"] == "How are you?"

    def test_only_tool_messages(self):
        conversation = [
            {"role": "tool", "content": "Result data"},
            {"role": "assistant", "content": "Done."},
        ]
        result = _extract_prompt(conversation)
        assert result == []


class TestGenerateResponses:
    def test_empty_prompt_returns_empty_response(self):
        """When prompt extraction yields nothing, response should be empty string."""
        test_samples = [
            {"conversation": [{"role": "tool", "content": "result"}]}
        ]
        mock_tokenizer = MagicMock()

        with (
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
        ):
            mock_model = MagicMock()
            mock_model.device = MagicMock()
            mock_model.generate.return_value = [[1, 2, 3]]
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            from src.evaluation.evaluate import _generate_responses

            predictions = _generate_responses(
                model_path="/fake/path",
                tokenizer=mock_tokenizer,
                test_samples=test_samples,
            )
            assert len(predictions) == 1
            assert predictions[0]["response"] == ""

    def test_generates_response_for_valid_prompt(self):
        test_samples = [
            {
                "conversation": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ]
            }
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "generated response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 5]
        mock_tokenizer.return_value = mock_batch

        with (
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
        ):
            mock_model = MagicMock()
            mock_model.device = MagicMock()
            mock_model.generate.return_value = [[1, 2, 3, 4, 5, 100, 101]]
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            from src.evaluation.evaluate import _generate_responses

            predictions = _generate_responses(
                model_path="/fake/path",
                tokenizer=mock_tokenizer,
                test_samples=test_samples,
            )
            assert len(predictions) == 1
            assert predictions[0]["response"] == "generated response"

    def test_lora_model_path(self):
        """Test that is_lora=True loads base + PeftModel correctly."""
        test_samples = [
            {
                "conversation": [
                    {"role": "user", "content": "Hi"},
                ]
            }
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "lora response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 3]
        mock_tokenizer.return_value = mock_batch

        with (
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("peft.PeftModel") as mock_peft_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
        ):
            mock_base = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_base

            mock_peft_model = MagicMock()
            mock_peft_model.device = MagicMock()
            mock_peft_model.generate.return_value = [[1, 2, 3, 99, 100]]
            mock_peft_cls.from_pretrained.return_value = mock_peft_model

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            from src.evaluation.evaluate import _generate_responses

            predictions = _generate_responses(
                model_path="/fake/lora",
                tokenizer=mock_tokenizer,
                test_samples=test_samples,
                base_model_path="/fake/base",
                is_lora=True,
            )
            assert len(predictions) == 1
            assert predictions[0]["response"] == "lora response"
            mock_model_cls.from_pretrained.assert_called_once()
            mock_peft_cls.from_pretrained.assert_called_once_with(
                mock_base, "/fake/lora"
            )

    def test_batch_processing(self):
        """Multiple samples should be processed correctly."""
        test_samples = [
            {"conversation": [{"role": "user", "content": "Q1"}]},
            {"conversation": [{"role": "user", "content": "Q2"}]},
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 4]
        mock_tokenizer.return_value = mock_batch

        with (
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
        ):
            mock_model = MagicMock()
            mock_model.device = MagicMock()
            mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            from src.evaluation.evaluate import _generate_responses

            predictions = _generate_responses(
                model_path="/fake/path",
                tokenizer=mock_tokenizer,
                test_samples=test_samples,
                batch_size=2,
            )
            assert len(predictions) == 2
            assert predictions[0]["response"] == "response"
            assert predictions[1]["response"] == "response"


class TestRunFullEvaluation:
    def test_run_base_model_only(self):
        """Only the base model path exists; SFT/GRPO don't."""
        mock_dataset_dict = {
            "train": [
                {"conversation": [{"role": "user", "content": "Hi"}]},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 3]
        mock_tokenizer.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 99]]

        mock_result = MagicMock()
        mock_result.average_reward = 0.5

        with (
            patch("src.evaluation.evaluate.os.makedirs") as mock_makedirs,
            patch("src.evaluation.evaluate.os.path.exists") as mock_exists,
            patch("src.evaluation.evaluate.load_dataset") as mock_load_dataset,
            patch(
                "src.evaluation.evaluate.AutoTokenizer"
            ) as mock_auto_tokenizer,
            patch(
                "src.evaluation.evaluate.AutoModelForCausalLM"
            ) as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
            patch("src.evaluation.evaluate.compute_metrics") as mock_metrics,
            patch(
                "src.evaluation.evaluate.print_comparison"
            ) as mock_comparison,
            patch("builtins.open") as mock_open_file,
        ):
            mock_load_dataset.return_value = mock_dataset_dict
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_exists.return_value = False  # No SFT/GRPO

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            mock_metrics.return_value = mock_result
            mock_comparison.return_value = "Comparison report text"

            from src.evaluation.evaluate import run_full_evaluation

            report = run_full_evaluation(
                test_data_path="/fake/test.json",
                output_dir="/fake/output",
            )

            assert report == "Comparison report text"
            mock_makedirs.assert_called_once_with("/fake/output", exist_ok=True)
            mock_load_dataset.assert_called_once_with(
                "json", data_files="/fake/test.json"
            )
            mock_auto_tokenizer.from_pretrained.assert_called_once()
            mock_model_cls.from_pretrained.assert_called_once()

    def test_run_with_sft_model(self):
        """SFT model adapter exists."""
        mock_dataset_dict = {
            "train": [
                {"conversation": [{"role": "user", "content": "Hi"}]},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 3]
        mock_tokenizer.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 99]]

        mock_result_base = MagicMock()
        mock_result_base.average_reward = 0.4
        mock_result_sft = MagicMock()
        mock_result_sft.average_reward = 0.7

        def metrics_side_effect(predictions, ground_truth, model_name):
            return mock_result_sft if "SFT" in model_name else mock_result_base

        with (
            patch("src.evaluation.evaluate.os.makedirs") as mock_makedirs,
            patch("src.evaluation.evaluate.os.path.exists") as mock_exists,
            patch("src.evaluation.evaluate.load_dataset") as mock_load_dataset,
            patch(
                "src.evaluation.evaluate.AutoTokenizer"
            ) as mock_auto_tokenizer,
            patch(
                "src.evaluation.evaluate.AutoModelForCausalLM"
            ) as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
            patch("src.evaluation.evaluate.compute_metrics") as mock_metrics,
            patch(
                "src.evaluation.evaluate.print_comparison"
            ) as mock_comparison,
            patch("builtins.open") as mock_open_file,
            patch("peft.PeftModel") as mock_peft_cls,
        ):
            mock_load_dataset.return_value = mock_dataset_dict
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            mock_base = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_base

            mock_peft_model = MagicMock()
            mock_peft_model.device = MagicMock()
            mock_peft_model.generate.return_value = [[1, 2, 3, 99]]
            mock_peft_cls.from_pretrained.return_value = mock_peft_model

            mock_exists.side_effect = (
                lambda p: "adapter_config.json" in p and "sft" in p.lower()
            )

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            mock_metrics.side_effect = metrics_side_effect
            mock_comparison.return_value = "Comparison with SFT"

            from src.evaluation.evaluate import run_full_evaluation

            report = run_full_evaluation(
                test_data_path="/fake/test.json",
                output_dir="/fake/output",
            )

            assert report == "Comparison with SFT"
            assert mock_comparison.call_count == 1

    def test_run_with_sft_full_model_and_grpo_adapter(self):
        """SFT exists as full model (not adapter) + GRPO adapter exists.
        Covers lines 154 (SFT full model) and 158 (GRPO adapter).
        """
        mock_dataset_dict = {
            "train": [
                {"conversation": [{"role": "user", "content": "Hi"}]},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 3]
        mock_tokenizer.return_value = mock_batch

        mock_base = MagicMock()
        mock_base.device = MagicMock()
        mock_base.generate.return_value = [[1, 2, 3, 99]]

        mock_sft = MagicMock()
        mock_sft.device = MagicMock()
        mock_sft.generate.return_value = [[1, 2, 3, 99]]

        mock_grpo_base = MagicMock()

        mock_grpo = MagicMock()
        mock_grpo.device = MagicMock()
        mock_grpo.generate.return_value = [[1, 2, 3, 99]]

        def model_side_effect(*args, **kwargs):
            model_path = args[0] if args else kwargs.get("pretrained_model_name_or_path", "")
            if "sft" in str(model_path).lower() and "adapter" not in str(model_path).lower():
                return mock_sft
            return mock_base

        def exists_side_effect(path):
            path_str = str(path)
            # SFT: adapter doesn't exist, full model does -> line 154
            if "sft" in path_str.lower() and "adapter_config" in path_str.lower():
                return False
            if "sft" in path_str.lower():
                return True
            # GRPO: adapter exists -> line 158
            if "grpo" in path_str.lower() and "adapter_config" in path_str.lower():
                return True
            return False

        mock_result_base = MagicMock()
        mock_result_base.average_reward = 0.4
        mock_result_sft = MagicMock()
        mock_result_sft.average_reward = 0.7
        mock_result_grpo = MagicMock()
        mock_result_grpo.average_reward = 0.9

        def metrics_side_effect(predictions, ground_truth, model_name):
            if "SFT" in model_name:
                return mock_result_sft
            if "GRPO" in model_name:
                return mock_result_grpo
            return mock_result_base

        with (
            patch("src.evaluation.evaluate.os.makedirs") as mock_makedirs,
            patch("src.evaluation.evaluate.os.path.exists") as mock_exists,
            patch("src.evaluation.evaluate.load_dataset") as mock_load_dataset,
            patch("src.evaluation.evaluate.AutoTokenizer") as mock_auto_tokenizer,
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
            patch("src.evaluation.evaluate.compute_metrics") as mock_metrics,
            patch("src.evaluation.evaluate.print_comparison") as mock_comparison,
            patch("builtins.open") as mock_open_file,
            patch("peft.PeftModel") as mock_peft_cls,
        ):
            mock_load_dataset.return_value = mock_dataset_dict
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.side_effect = model_side_effect
            mock_exists.side_effect = exists_side_effect
            mock_peft_cls.from_pretrained.return_value = mock_grpo

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            mock_metrics.side_effect = metrics_side_effect
            mock_comparison.return_value = "Comparison with SFT full + GRPO adapter"

            from src.evaluation.evaluate import run_full_evaluation

            report = run_full_evaluation(
                test_data_path="/fake/test.json",
                output_dir="/fake/output",
            )

            assert report == "Comparison with SFT full + GRPO adapter"
            mock_peft_cls.from_pretrained.assert_called_once()

    def test_run_with_grpo_full_model(self):
        """GRPO exists as full model (not adapter) -> covers line 162."""
        mock_dataset_dict = {
            "train": [
                {"conversation": [{"role": "user", "content": "Hi"}]},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "response"

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_batch.input_ids.shape = [1, 3]
        mock_tokenizer.return_value = mock_batch

        mock_base = MagicMock()
        mock_base.device = MagicMock()
        mock_base.generate.return_value = [[1, 2, 3, 99]]

        mock_grpo = MagicMock()
        mock_grpo.device = MagicMock()
        mock_grpo.generate.return_value = [[1, 2, 3, 99]]

        def model_side_effect(*args, **kwargs):
            model_path = args[0] if args else kwargs.get("pretrained_model_name_or_path", "")
            if "grpo" in str(model_path).lower():
                return mock_grpo
            return mock_base

        def exists_side_effect(path):
            path_str = str(path)
            # SFT doesn't exist at all
            if "sft" in path_str.lower():
                return False
            # GRPO: adapter doesn't exist, full model does -> line 162
            if "grpo" in path_str.lower() and "adapter_config" in path_str.lower():
                return False
            if "grpo" in path_str.lower():
                return True
            return False

        mock_result_base = MagicMock()
        mock_result_base.average_reward = 0.4
        mock_result_grpo = MagicMock()
        mock_result_grpo.average_reward = 0.9

        def metrics_side_effect(predictions, ground_truth, model_name):
            if "GRPO" in model_name:
                return mock_result_grpo
            return mock_result_base

        with (
            patch("src.evaluation.evaluate.os.makedirs") as mock_makedirs,
            patch("src.evaluation.evaluate.os.path.exists") as mock_exists,
            patch("src.evaluation.evaluate.load_dataset") as mock_load_dataset,
            patch("src.evaluation.evaluate.AutoTokenizer") as mock_auto_tokenizer,
            patch("src.evaluation.evaluate.AutoModelForCausalLM") as mock_model_cls,
            patch("src.evaluation.evaluate.torch") as mock_torch,
            patch("src.evaluation.evaluate.compute_metrics") as mock_metrics,
            patch("src.evaluation.evaluate.print_comparison") as mock_comparison,
            patch("builtins.open") as mock_open_file,
        ):
            mock_load_dataset.return_value = mock_dataset_dict
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.side_effect = model_side_effect
            mock_exists.side_effect = exists_side_effect

            mock_no_grad = MagicMock()
            mock_no_grad.__enter__.return_value = None
            mock_no_grad.__exit__.return_value = None
            mock_torch.no_grad.return_value = mock_no_grad

            mock_metrics.side_effect = metrics_side_effect
            mock_comparison.return_value = "Comparison with GRPO full model"

            from src.evaluation.evaluate import run_full_evaluation

            report = run_full_evaluation(
                test_data_path="/fake/test.json",
                output_dir="/fake/output",
            )

            assert report == "Comparison with GRPO full model"
