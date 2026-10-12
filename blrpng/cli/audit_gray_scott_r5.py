"""Final provenance/storage audit for the R5 Gray-Scott confirmation."""
from __future__ import annotations

import json
from pathlib import Path

from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.gray_scott_r5_safe_router import PROTOCOL, protocol
from blrpng.data.capacity import artifact_size_bytes
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import (git_status_outside_project, resolve_in_project,
                          sha256_file, write_json)


def main():
    with workspace_audit('audit_gray_scott_r5'):
        p = protocol()
        summary_path = 'artifacts/reports/gray_scott_r5/summary.json'
        summary = read(summary_path)
        checked_checkpoints = {}
        for task, spec in p['tasks'].items():
            for source, candidate in spec['candidates'].items():
                if sha256_file(candidate['checkpoint']) != candidate['sha256']:
                    raise ValueError('R5 candidate checkpoint changed')
                checked_checkpoints[f'task{task}/{source}'] = candidate['sha256']
        query_hashes, pairings = {}, {}
        for resolution in (128, 256):
            pairings[resolution] = {}
            for task in p['tasks']:
                path = f'artifacts/data/gray_scott_r5/task{int(task):02d}_query_{resolution}.index.json'
                index = read_index(path, verify_hashes=True)
                if index['metadata']['protocol_sha256'] != p['sha256']:
                    raise ValueError('R5 index protocol mismatch')
                rows = index['shards']
                if not all(r['solver_converged'] and r['final_residual_max'] <= 1e-8 for r in rows):
                    raise ValueError('R5 solver gate failed')
                pairings[resolution][task] = sorted((r['pair_id'], r['initial_condition_id'], r['boundary_id'])
                                                    for r in rows)
                query_hashes[path] = sha256_file(path)
        if pairings[128] != pairings[256]:
            raise ValueError('R5 cross-resolution pairing mismatch')
        figure_manifest = read('artifacts/figures/gray_scott_r5/manifest.json')
        for figure in figure_manifest['figures']:
            if sha256_file(figure['png']) != figure['png_sha256']:
                raise ValueError('R5 figure hash mismatch')
        if sha256_file(figure_manifest['csv']) != figure_manifest['csv_sha256']:
            raise ValueError('R5 metric table hash mismatch')
        outside = git_status_outside_project()
        if outside:
            raise RuntimeError(f'Unexpected outside-project worktree changes: {outside}')
        size = artifact_size_bytes()
        if size > 300_000_000_000:
            raise RuntimeError('Artifact storage exceeds 300 GB')
        assessment = summary['stage_assessment']
        result = {
            'stage_completed': True,
            'safe_router_point_gate_passed': assessment['practical_stability_gate_passed'],
            'safe_router_statistical_superiority_confirmed': assessment['statistical_superiority_confirmed'],
            'newton_mechanism_claim_passed': False,
            'seed': 42,
            'scope': summary['scope'],
            'route': summary['route'],
            'assessment': assessment,
            'limitations': summary['limitations'],
            'protocol_sha256': p['sha256'],
            'protocol_file_sha256': sha256_file(PROTOCOL),
            'summary_sha256': sha256_file(summary_path),
            'query_index_sha256': query_hashes,
            'checked_checkpoint_sha256': checked_checkpoints,
            'figure_manifest_sha256': sha256_file('artifacts/figures/gray_scott_r5/manifest.json'),
            'png_count': len(figure_manifest['figures']),
            'artifact_bytes': size,
            'outside_project_status': outside,
            'code_sha256': {path: sha256_file(path) for path in (
                'blrpng/cli/gray_scott_r5_safe_router.py',
                'blrpng/cli/finalize_gray_scott_r5.py',
                'blrpng/cli/audit_gray_scott_r5.py',
                'blrpng/cli/diagnose_gray_scott_multires_consistency.py',
                'blrpng/training/gray_scott_commutator.py',
                'blrpng/training/gray_scott_r3.py')},
            'next_decision': ('Use the safe router as the practical Gray-Scott result, but test Newton-specific '
                              'mechanism only in a prescreened nonlinear regime where Newton corrections '
                              'materially differ from the endpoint increment.'),
        }
        write_json('artifacts/reports/gray_scott_r5/stage_conclusion.json', result)
        percentage = lambda value: f'{100 * value:.2f}%'
        markdown = f'''# Gray–Scott R5 阶段结论

R5 使用五个既有开发边界、每个边界两个**冻结的新初值**，在 128²/256² 上确认了
预先冻结的 safe-router。路由只读取 64² 有标签验证误差和无标签 free20 128→256
一致性，不读取任何 R5 高分辨率真实解。仅 seed42；这不是新边界或多随机种子正式结论。

## 256²、50步两场等权平均误差

- 始终选 ordinary：{percentage(assessment['always_final_h50_balanced'])}
- 始终选 Newton path：{percentage(assessment['always_path_h50_balanced'])}
- safe-router：{percentage(assessment['safe_router_h50_balanced'])}
- router 相对 ordinary 改善：{percentage(assessment['router_improvement_vs_always_final'])}
- router 相对 path 改善：{percentage(assessment['router_improvement_vs_always_path'])}

冻结路由为：{json.dumps(summary['route'], ensure_ascii=False)}。task16 的 path 虽然 64²
验证误差更低，但 free20 128→256 不一致性为 21.99%，超过 6% 阈值，因此被拒绝。

## 科学结论边界

safe-router 的独立新初值确认门控为 **{assessment['practical_stability_gate_passed']}**。
但相对 ordinary/path 的条件 bootstrap 95% CI 分别为
{assessment['router_vs_final_conditional_ci95']} 和 {assessment['router_vs_path_conditional_ci95']}，
均跨零，因此统计优越性为 **{assessment['statistical_superiority_confirmed']}**；当前只能报告点估计和反例规避，
不能报告已统计确证的普遍增益。
该结果在 task16 的新初值上复现了“无标签长时跨分辨率诊断可规避灾难性失稳”；它不证明
Newton path 在当前 Gray–Scott 设置具有独立机制优势。当前 dt=1 数据里累计 Newton
correction 与 endpoint increment 的平均差只有约 0.021%，而 endpoint 辅助对照与
Newton 辅助对照近似等价，所以 Newton-specific mechanism gate 仍为 **False**。

下一步只在“Newton correction 与 endpoint increment 存在实质差异”的更强非线性小型
预筛选设置中检验路径机制；若预筛选不过，直接转向具有丰富 Newton 迭代的超弹性，
避免为一个信息退化的 Gray–Scott 路径标签继续扩大数据。

PNG 位于 `artifacts/figures/gray_scott_r5/`；每张图均附来源哈希，绘图未重新选择模型。
'''
        output = resolve_in_project('artifacts/reports/gray_scott_r5/阶段结论.md')
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(markdown, encoding='utf-8')
        print(json.dumps({'completed': True, 'safe_router_point_gate_passed': result['safe_router_point_gate_passed'],
                          'statistical_superiority': result['safe_router_statistical_superiority_confirmed'],
                          'pngs': result['png_count'], 'artifact_gb': size / 1e9}), flush=True)


if __name__ == '__main__':
    main()
