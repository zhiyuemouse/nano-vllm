import json
import statistics
from pathlib import Path
import math
import copy

def summarize_trials(trials: list[dict]) -> dict:
    """
    每个请求分别返回：
        ttft_median_ms
        itl1_median_ms
        itl2_median_ms
        itl3_median_ms
        request_latency_median_ms
    """
    tmp = {}
    for c in ['A', 'B', 'C']:
        tmp[c] = {
            "ttft_list": [],
            "itl1_list": [],
            "itl2_list": [],
            "itl3_list": [],
            "request_latency_list": [],
        }

    for epoch_dict in trials:
        for c in ['A', 'B', 'C']:
            tmp[c]["ttft_list"].append(epoch_dict[c]["ttft_ms"])
            tmp[c]["itl1_list"].append(epoch_dict[c]["itls_ms"][0])
            tmp[c]["itl2_list"].append(epoch_dict[c]["itls_ms"][1])
            tmp[c]["itl3_list"].append(epoch_dict[c]["itls_ms"][2])
            tmp[c]["request_latency_list"].append(
                epoch_dict[c]["ttft_ms"] + epoch_dict[c]["itls_ms"][0] + epoch_dict[c]["itls_ms"][1] + epoch_dict[c]["itls_ms"][2]
            )
    output = {}
    for c in ['A', 'B', 'C']:
        output[c] = dict()
        output[c]["ttft_median_ms"] = statistics.median(tmp[c]["ttft_list"])
        output[c]["itl1_median_ms"] = statistics.median(tmp[c]["itl1_list"])
        output[c]["itl2_median_ms"] = statistics.median(tmp[c]["itl2_list"])
        output[c]["itl3_median_ms"] = statistics.median(tmp[c]["itl3_list"])
        output[c]["request_latency_median_ms"] = statistics.median(tmp[c]["request_latency_list"])
    return output

def test_summarize_trials():
    # 五轮相同数据，覆盖 A/B/C 三个请求。
    trials = [
        {
            label: {
                "ttft_ms": 200.0,
                "itls_ms": [100.0, 200.0, 0.0],
            }
            for label in ["A", "B", "C"]
        }
        for _ in range(5)
    ]

    actual = summarize_trials(trials)

    expected = {
        "ttft_median_ms": 200.0,
        "itl1_median_ms": 100.0,
        "itl2_median_ms": 200.0,
        "itl3_median_ms": 0.0,
        "request_latency_median_ms": 500.0,
    }

    assert set(actual) == {"A", "B", "C"}

    for label in ["A", "B", "C"]:
        for metric, expected_value in expected.items():
            assert math.isclose(
                actual[label][metric],
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ), (
                f"{label} 的 {metric} 错误："
                f"实际 {actual[label][metric]}，预期 {expected_value}"
            )

    print("test_summarize_trials PASS")

def validate_comparison(payload_0: dict, payload_2: dict) -> None:
    """配置不符合本次对照约定时，抛出 ValueError。"""
    if payload_0["metadata"]['policy_limit'] != 0: raise ValueError("metadata.policy_limit 不匹配")
    if payload_2["metadata"]['policy_limit'] != 2: raise ValueError("metadata.policy_limit 不匹配")
    if payload_0["metadata"]['llm_config']['max_consecutive_prefills'] != payload_0["metadata"]['policy_limit']: raise ValueError("llm_config.max_consecutive_prefills 与上述值不一致")
    if payload_2["metadata"]['llm_config']['max_consecutive_prefills'] != payload_2["metadata"]['policy_limit']: raise ValueError("llm_config.max_consecutive_prefills 与上述值不一致")
    config_0 = payload_0["metadata"]["llm_config"].copy()
    config_2 = payload_2["metadata"]["llm_config"].copy()

    config_0.pop("max_consecutive_prefills")
    config_2.pop("max_consecutive_prefills")

    if config_0 != config_2:
        raise ValueError("llm_config 不一致")

    if payload_0["metadata"]['warmup'] != payload_2["metadata"]['warmup']: raise ValueError("warmup 不一致")
    if payload_0["metadata"]['repeats'] != payload_2["metadata"]['repeats']: raise ValueError("repeats 不一致")
    if payload_0["metadata"]['seed'] != payload_2["metadata"]['seed']: raise ValueError("seed 不一致")
    if payload_0["metadata"]['arrival_mode'] != payload_2["metadata"]['arrival_mode']: raise ValueError("arrival_mode 不一致")
    if payload_0["metadata"]['git_commit'] != payload_2["metadata"]['git_commit']: raise ValueError("git_commit 不一致")
    if payload_0["metadata"]['gpu_name'] != payload_2["metadata"]['gpu_name']: raise ValueError("gpu_name 不一致")
    if payload_0["metadata"]['versions'] != payload_2["metadata"]['versions']: raise ValueError("versions 不一致")
    if payload_0["metadata"]['model_path'] != payload_2["metadata"]['model_path']: raise ValueError("model_path 不一致")
    if payload_0["metadata"]['prompt_token_ids'] != payload_2["metadata"]['prompt_token_ids']: raise ValueError("prompt_token_ids 不一致")
    if payload_0["metadata"]['sampling_config'] != payload_2["metadata"]['sampling_config']: raise ValueError("sampling_config 不一致")
    print(f"开始时间: {payload_0['metadata']['started_at']} | payload_0 git_dirty : {payload_0['metadata']['git_dirty']}")
    print(f"开始时间: {payload_2['metadata']['started_at']} | payload_2 git_dirty : {payload_2['metadata']['git_dirty']}")
    for payload in [payload_0, payload_2]:
        trials = payload["trials"]
        metadata = payload["metadata"]

        if not (len(trials) == metadata["repeats"] == 5): raise ValueError("不满足 len(trials) == metadata['repeats'] == 5")

        for trial in trials:
            if not (set(trial) == {"A", "B", "C"}): raise ValueError("不满足 set(trial) == {'A', 'B', 'C'}")

            for metrics in trial.values():
                if not (len(metrics["itls_ms"]) == metadata["sampling_config"]["max_tokens"] - 1): raise ValueError("不满足 len(metrics['itls_ms']) == metadata['sampling_config']['max_tokens'] - 1")

                for value in [metrics["ttft_ms"], *metrics["itls_ms"]]:
                    if not (math.isfinite(value) and value >= 0): raise ValueError("不满足 math.isfinite(value) and value >= 0")


def compare_scheduler_policies():
    output_dir = Path(__file__).resolve().parent
    with (output_dir / "results" / "policy_0_run1.json").open("r", encoding="utf-8") as f:
        policy_0 = json.load(f)
    with (output_dir / "results" / "policy_2_run1.json").open("r", encoding="utf-8") as f:
        policy_2 = json.load(f)

    validate_comparison(policy_0, policy_2)
    policy_0_bad = copy.deepcopy(policy_0)
    policy_0_bad["metadata"]["seed"] = 89
    try:
        validate_comparison(policy_0_bad, policy_2)
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")

    policy_0_output = summarize_trials(policy_0["trials"])
    policy_2_output = summarize_trials(policy_2["trials"])
    for c in ["A", "B", "C"]:
        for metric in ["ttft_median_ms", "itl1_median_ms", "itl2_median_ms", "itl3_median_ms", "request_latency_median_ms"]:
            print(f"请求:{c} | 指标:{metric} | 参数0中位数:{policy_0_output[c][metric]:.2f} | ",
                  f"参数2中位数:{policy_2_output[c][metric]:.2f} | ",
                  f"变化率:{(policy_2_output[c][metric] - policy_0_output[c][metric]) / policy_0_output[c][metric] * 100:.2f}")

if __name__ == "__main__":
    test_summarize_trials()
    compare_scheduler_policies()