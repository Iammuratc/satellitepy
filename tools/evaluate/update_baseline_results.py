#!/usr/bin/env python3
"""Refresh the consolidated YOLO baseline results spreadsheet (CSV)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path('/output/yolo11_baselines')
OUTPUT = Path('/workspace/satellitepy/docs/baseline_results.csv')
PROJECT = 'satellitepy/yolo11-multidataset-baselines'

EXPERIMENTS = (
    ('FineAir', 'joint', 'semantic gradients update backbone',
     'fineair-y11s-sep-s0', ROOT / 'separate/fineair_seed0'),
    ('FineAir', 'detached', 'semantic gradients stopped at feature input',
     'fineair-y11s-det-s0', ROOT / 'detached/fineair_seed0'),
    ('FineAir', 'detection_only', 'no semantic heads',
     'fineair-y11s-detonly-s0', ROOT / 'detection_only/fineair_seed0'),
    ('FineAir', 'detection_only_seed1', 'independent detector; no semantic heads; seed 1',
     'fineair-y11s-detonly-s1', ROOT / 'detection_only/fineair_seed1'),
    ('FineAir', 'detection_only_seed2', 'independent detector; no semantic heads; seed 2',
     'fineair-y11s-detonly-s2', ROOT / 'detection_only/fineair_seed2'),
    ('RarePlanes', 'joint', 'semantic gradients update backbone',
     'rareplanes-y11s-joint-s0', ROOT / 'rareplanes/joint/seed0'),
    ('RarePlanes', 'detached', 'semantic gradients stopped at feature input',
     'rareplanes-y11s-det-s0', ROOT / 'rareplanes/detached/seed0'),
    ('RarePlanes', 'detection_only', 'no semantic heads',
     'rareplanes-y11s-detonly-s0', ROOT / 'rareplanes/detection_only/seed0'),
    ('RarePlanes', 'detection_only_seed1',
     'independent detector; no semantic heads; seed 1',
     'rareplanes-y11s-detonly-s1', ROOT / 'rareplanes/detection_only/seed1'),
    ('RarePlanes', 'detection_only_seed2',
     'independent detector; no semantic heads; seed 2',
     'rareplanes-y11s-detonly-s2', ROOT / 'rareplanes/detection_only/seed2'),
    ('DOTA-ships', 'detection_only', 'independent ship detector; RGB',
     'dota-ships-y11s-detonly-s0', ROOT / 'dota_ships/detection_only/seed0'),
    ('DOTA-aircraft', 'detection_only', 'independent aircraft detector; RGB',
     'dota-aircraft-y11s-detonly-s0', ROOT / 'dota_aircraft_controls/independent/seed0'),
    ('DOTA-aircraft', 'detection_masks',
     'independent aircraft detector with joint foreground-mask auxiliary; RGB',
     'dota-aircraft-y11s-detection-masks-s0',
     ROOT / 'dota_aircraft_controls/independent_detection_masks/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'strict_detached',
     'one aircraft head; shared backbone/FPN; detached semantics',
     'aircraft-dota-aircraft-y11s-strict-det-s0',
     ROOT / 'dota_aircraft_controls/strict_detached/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'pcgrad',
     'strict-detached model; PCGrad across per-dataset detection objectives',
     'aircraft-dota-aircraft-y11s-pcgrad-s0',
     ROOT / 'dota_aircraft_controls/pcgrad/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'early_dsbn',
     'shared convolutions; dataset-specific BatchNorm in backbone modules 0-3; detached semantics',
     'aircraft-dota-aircraft-y11s-early-dsbn-s0',
     ROOT / 'dota_aircraft_controls/early_dsbn/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'detection_gated',
     'one aircraft head; detached semantics; detection-trained residual gates',
     'aircraft-dota-aircraft-y11s-detection-gated-s0',
     ROOT / 'dota_aircraft_controls/detection_gated/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'domain_adapters',
     'shared aircraft head; dataset-specific gated P3/P4/P5 residual adapters',
     'aircraft-dota-aircraft-y11s-domain-adapters-s0',
     ROOT / 'dota_aircraft_controls/domain_adapters/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter',
     'shared identity path for aircraft datasets; gated residual adapter for DOTA only',
     'aircraft-dota-aircraft-y11s-dota-adapter-s0',
     ROOT / 'dota_aircraft_controls/dota_adapter/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter_early_dsbn',
     'early dataset-specific BatchNorm plus gated P3/P4/P5 residual adapter for DOTA only',
     'aircraft-dota-aircraft-y11s-dota-adapter-early-dsbn-s0',
     ROOT / 'dota_aircraft_controls/dota_adapter_early_dsbn/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter_freeze_early',
     'DOTA-only adapter; backbone modules 1-3 and their BN statistics frozen; input stem trainable',
     'aircraft-dota-aircraft-y11s-dota-adapter-freeze-early-s0',
     ROOT / 'dota_aircraft_controls/dota_adapter_freeze_early/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter_masks_isolated',
     'DOTA adapter and mask head share mask gradients; mask gradient detached from shared FPN',
     'aircraft-dota-aircraft-y11s-dota-adapter-masks-isolated-s0',
     ROOT / 'dota_aircraft_controls/dota_adapter_masks_isolated/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_fpn',
     'shared early backbone and aircraft head; DOTA-specific FPN/neck',
     'aircraft-dota-aircraft-y11s-dota-fpn-s0',
     ROOT / 'dota_aircraft_controls/dota_fpn/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_late_backbone_fpn',
     'shared backbone modules 0-7 and aircraft head; DOTA-specific late P5 backbone and FPN',
     'aircraft-dota-aircraft-y11s-late-backbone-fpn-s0',
     ROOT / 'dota_aircraft_controls/dota_late_backbone_fpn/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_late_backbone_fpn_masks',
     'DOTA-specific late P5 backbone/FPN with DOTA-only aircraft-mask auxiliary loss',
     'aircraft-dota-aircraft-y11s-late-backbone-fpn-masks-s0',
     ROOT / 'dota_aircraft_controls/dota_late_backbone_fpn_masks/seed0'),
    ('RarePlanes→FineAir', 'sequential_detection_only',
     'RarePlanes weights followed by FineAir-only optimization',
     'sequential-rareplanes-to-fineair-y11s-s0',
     ROOT / 'sequential_transfer/rareplanes_to_fineair/seed0'),
    ('FineAir→RarePlanes', 'sequential_detection_only',
     'FineAir weights followed by RarePlanes-only optimization',
     'sequential-fineair-to-rareplanes-y11s-s0',
     ROOT / 'sequential_transfer/fineair_to_rareplanes/seed0'),
    ('FineAir+RarePlanes+DOTA-ships', 'conditional_detection_only',
     'shared backbone/FPN; conditionally supervised aircraft and ship OBB heads',
     'aircraft-dota-ships-y11s-conditional-detonly-s0',
     ROOT / 'aircraft_dota_ships/conditional_detection_only/seed0'),
    ('FineAir+RarePlanes+DOTA-ships', 'conditional_joint',
     'conditional aircraft/ship OBB heads; all semantic gradients shared',
     'aircraft-dota-ships-y11s-conditional-joint-s0',
     ROOT / 'aircraft_dota_ships/conditional_joint/seed0'),
    ('FineAir+RarePlanes+DOTA-ships', 'conditional_strict_detached',
     'conditional aircraft/ship OBB heads; detached semantics with isolated optimization',
     'aircraft-dota-ships-y11s-conditional-strict-det-s0',
     ROOT / 'aircraft_dota_ships/conditional_strict_detached/seed0'),
    ('FineAir+RarePlanes+DOTA-ships', 'conditional_detection_gated',
     'detached semantics; detection-trained aircraft/ship residual gates',
     'aircraft-dota-ships-y11s-conditional-detection-gated-s0',
     ROOT / 'aircraft_dota_ships/conditional_detection_gated/seed0'),
    ('FineAir+RarePlanes+DOTA-ships', 'conditional_detection_gated_rebalanced',
     'warm-started gated model; fresh optimizer; 25/25/50 dataset sampling',
     'aircraft-dota-ships-y11s-detection-gated-rebalanced-s0',
     ROOT / 'aircraft_dota_ships/conditional_detection_gated_rebalanced/seed0'),
    ('FineAir+RarePlanes', 'joint', 'one shared backbone; all semantic gradients enabled',
     'fineair-rareplanes-y11s-joint-s0', ROOT / 'fineair_rareplanes/joint/seed0'),
    ('FineAir+RarePlanes', 'detached', 'one shared backbone; semantic gradients detached',
     'fineair-rareplanes-y11s-det-s0', ROOT / 'fineair_rareplanes/detached/seed0'),
    ('FineAir+RarePlanes', 'detection_only', 'one shared backbone; no semantic heads',
     'fineair-rareplanes-y11s-detonly-s0', ROOT / 'fineair_rareplanes/detection_only/seed0'),
    ('FineAir+RarePlanes', 'learned_gates',
     'detached semantic base plus learned task residual gates',
     'fineair-rareplanes-y11s-gated-s0', ROOT / 'fineair_rareplanes/gated/seed0'),
    ('FineAir+RarePlanes', 'joint_corrected',
     'joint gradients; corrected full per-dataset validation protocol',
     'fineair-rareplanes-y11s-joint-corrected-s0',
     ROOT / 'fineair_rareplanes/joint_corrected/seed0'),
    ('FineAir+RarePlanes', 'strict_detached',
     'detached semantics; disjoint optimizers and independent AMP scalers',
     'fineair-rareplanes-y11s-strict-det-s0',
     ROOT / 'fineair_rareplanes/strict_detached/seed0'),
    ('FineAir+RarePlanes', 'learned_gates_balanced',
     'learned task residual gates; equal-probability dataset sampling',
     'fineair-rareplanes-y11s-gated-balanced-s0',
     ROOT / 'fineair_rareplanes/gated_balanced/seed0'),
    ('FineAir+RarePlanes', 'detection_only_balanced',
     'no semantic heads; equal-probability dataset sampling',
     'fineair-rareplanes-y11s-detonly-balanced-s0',
     ROOT / 'fineair_rareplanes/detection_only_balanced/seed0'),
    ('DOTA-aircraft', 'detection_only_seed1', 'independent aircraft detector; RGB; seed 1',
     'dota-aircraft-y11s-detonly-s1', ROOT / 'dota_aircraft_controls/independent/seed1'),
    ('DOTA-aircraft', 'detection_only_seed2', 'independent aircraft detector; RGB; seed 2',
     'dota-aircraft-y11s-detonly-s2', ROOT / 'dota_aircraft_controls/independent/seed2'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'strict_detached_seed1',
     'shared backbone/FPN; detached semantics; seed 1',
     'aircraft-dota-aircraft-y11s-strict-det-s1',
     ROOT / 'dota_aircraft_controls/strict_detached/seed1'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'strict_detached_seed2',
     'shared backbone/FPN; detached semantics; seed 2',
     'aircraft-dota-aircraft-y11s-strict-det-s2',
     ROOT / 'dota_aircraft_controls/strict_detached/seed2'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter_seed1',
     'DOTA-only gated residual adapter; seed 1',
     'aircraft-dota-aircraft-y11s-dota-adapter-s1',
     ROOT / 'dota_aircraft_controls/dota_adapter/seed1'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dota_adapter_seed2',
     'DOTA-only gated residual adapter; seed 2',
     'aircraft-dota-aircraft-y11s-dota-adapter-s2',
     ROOT / 'dota_aircraft_controls/dota_adapter/seed2'),
    ('DOTA-aircraft-GSD030', 'detection_only',
     'independent detector; native DOTA imagery coarser than 0.30 m/px bicubic-upscaled before patching',
     'dota-aircraft-gsd030-y11s-detonly-s0',
     ROOT / 'dota_gsd030_controls/independent/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft-GSD030', 'dota_adapter',
     'shared detector with DOTA-only adapter; coarse DOTA imagery bicubic-upscaled to at most 0.30 m/px',
     'aircraft-dota-aircraft-gsd030-y11s-dota-adapter-s0',
     ROOT / 'dota_gsd030_controls/dota_adapter/seed0'),
    ('DOTA-aircraft-FDA', 'detection_only',
     'independent detector; training-only FDA from balanced FineAir/RarePlanes RGB references',
     'dota-aircraft-fda-y11s-detonly-s0',
     ROOT / 'dota_fda_controls/independent/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft-FDA', 'dota_adapter',
     'shared DOTA adapter; training-only FDA beta=0.01; native DOTA validation/test',
     'aircraft-dota-aircraft-fda-y11s-dota-adapter-s0',
     ROOT / 'dota_fda_controls/dota_adapter/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'early_backbone_routes',
     'dataset-specific modules 0-4; shared late backbone, FPN, and aircraft head',
     'aircraft-dota-aircraft-y11s-early-backbone-routes-s0',
     ROOT / 'dota_aircraft_controls/early_backbone_routes/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'early_backbone_routes_seed1',
     'dataset-specific modules 0-4; shared late backbone, FPN, and aircraft head; seed 1',
     'aircraft-dota-aircraft-y11s-early-backbone-routes-s1',
     ROOT / 'dota_aircraft_controls/early_backbone_routes/seed1'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'conditional_aircraft_heads',
     'shared backbone/FPN; FineAir+RarePlanes head and separate DOTA-aircraft head',
     'aircraft-dota-aircraft-y11s-conditional-heads-s0',
     ROOT / 'dota_aircraft_controls/conditional_heads/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'conditional_aircraft_heads_seed1',
     'shared backbone/FPN; FineAir+RarePlanes head and separate DOTA-aircraft head; seed 1',
     'aircraft-dota-aircraft-y11s-conditional-heads-s1',
     ROOT / 'dota_aircraft_controls/conditional_heads/seed1'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dynamic_dota_adapter',
     'DOTA-only spatial bottleneck expert with per-image channel gates at P3/P4/P5',
     'aircraft-dota-aircraft-y11s-dynamic-adapter-s0',
     ROOT / 'dota_aircraft_controls/dynamic_dota_adapter/seed0'),
    ('FineAir+RarePlanes+DOTA-aircraft', 'dynamic_dota_adapter_seed1',
     'DOTA-only spatial bottleneck expert with per-image channel gates at P3/P4/P5; seed 1',
     'aircraft-dota-aircraft-y11s-dynamic-adapter-s1',
     ROOT / 'dota_aircraft_controls/dynamic_dota_adapter/seed1'),
)

TEST_RESULTS = {
    'fineair-y11s-sep-s0': {
        'fineair': ROOT / 'evaluations/fineair_joint_seed0_test.json'},
    'fineair-y11s-det-s0': {
        'fineair': ROOT / 'evaluations/fineair_detached_seed0_test.json'},
    'fineair-y11s-detonly-s0': {
        'fineair': ROOT / 'evaluations/fineair_detection_only_seed0_test.json'},
    'fineair-y11s-detonly-s1': {
        'fineair': ROOT / 'evaluations/fineair_detection_only_seed1_test.json'},
    'fineair-y11s-detonly-s2': {
        'fineair': ROOT / 'evaluations/fineair_detection_only_seed2_test.json'},
    'rareplanes-y11s-joint-s0': {
        'rareplanes': ROOT / 'evaluations/rareplanes_joint_seed0_test.json'},
    'rareplanes-y11s-det-s0': {
        'rareplanes': ROOT / 'evaluations/rareplanes_detached_seed0_test.json'},
    'rareplanes-y11s-detonly-s0': {
        'rareplanes': ROOT / 'evaluations/rareplanes_detection_only_seed0_test.json'},
    'rareplanes-y11s-detonly-s1': {
        'rareplanes': ROOT / 'evaluations/rareplanes_detection_only_seed1_test.json'},
    'rareplanes-y11s-detonly-s2': {
        'rareplanes': ROOT / 'evaluations/rareplanes_detection_only_seed2_test.json'},
    'dota-ships-y11s-detonly-s0': {
        'dota_ships': ROOT / 'evaluations/dota_ships_detection_only_seed0_test.json'},
    'dota-aircraft-y11s-detonly-s0': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_detection_only_seed0_test.json'},
    'dota-aircraft-y11s-detection-masks-s0': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_detection_masks_seed0_test.json'},
    'aircraft-dota-aircraft-y11s-strict-det-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-pcgrad-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_pcgrad_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_pcgrad_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_pcgrad_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-early-dsbn-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_early_dsbn_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_early_dsbn_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_early_dsbn_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-detection-gated-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_detection_gated_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_detection_gated_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_detection_gated_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-domain-adapters-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_domain_adapters_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_domain_adapters_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_domain_adapters_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-early-dsbn-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_early_dsbn_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_early_dsbn_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_early_dsbn_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-freeze-early-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_freeze_early_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_freeze_early_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_freeze_early_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-masks-isolated-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_masks_isolated_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_masks_isolated_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_masks_isolated_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-fpn-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_fpn_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_fpn_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_fpn_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-late-backbone-fpn-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-late-backbone-fpn-masks-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_masks_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_masks_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_late_backbone_fpn_masks_dota_aircraft_test.json'},
    'sequential-rareplanes-to-fineair-y11s-s0': {
        'fineair': ROOT / 'evaluations/sequential_rareplanes_to_fineair_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/sequential_rareplanes_to_fineair_rareplanes_test.json'},
    'sequential-fineair-to-rareplanes-y11s-s0': {
        'fineair': ROOT / 'evaluations/sequential_fineair_to_rareplanes_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/sequential_fineair_to_rareplanes_rareplanes_test.json'},
    'aircraft-dota-ships-y11s-conditional-detonly-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_ships_detection_only_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_ships_detection_only_rareplanes_test.json',
        'dota_ships': ROOT / 'evaluations/aircraft_dota_ships_detection_only_dota_ships_test.json'},
    'aircraft-dota-ships-y11s-conditional-joint-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_ships_joint_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_ships_joint_rareplanes_test.json',
        'dota_ships': ROOT / 'evaluations/aircraft_dota_ships_joint_dota_ships_test.json'},
    'aircraft-dota-ships-y11s-conditional-strict-det-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_ships_strict_detached_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_ships_strict_detached_rareplanes_test.json',
        'dota_ships': ROOT / 'evaluations/aircraft_dota_ships_strict_detached_dota_ships_test.json'},
    'aircraft-dota-ships-y11s-conditional-detection-gated-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_rareplanes_test.json',
        'dota_ships': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_dota_ships_test.json'},
    'aircraft-dota-ships-y11s-detection-gated-rebalanced-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_rebalanced_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_rebalanced_rareplanes_test.json',
        'dota_ships': ROOT / 'evaluations/aircraft_dota_ships_detection_gated_rebalanced_dota_ships_test.json'},
    'fineair-rareplanes-y11s-joint-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_joint_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_joint_rareplanes_test.json'},
    'fineair-rareplanes-y11s-det-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_detached_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_detached_rareplanes_test.json'},
    'fineair-rareplanes-y11s-detonly-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_detection_only_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_detection_only_rareplanes_test.json'},
    'fineair-rareplanes-y11s-gated-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_gated_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_gated_rareplanes_test.json'},
    'fineair-rareplanes-y11s-joint-corrected-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_joint_corrected_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_joint_corrected_rareplanes_test.json'},
    'fineair-rareplanes-y11s-strict-det-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_strict_detached_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_strict_detached_rareplanes_test.json'},
    'fineair-rareplanes-y11s-gated-balanced-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_gated_balanced_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_gated_balanced_rareplanes_test.json'},
    'fineair-rareplanes-y11s-detonly-balanced-s0': {
        'fineair': ROOT / 'evaluations/fineair_rareplanes_detection_only_balanced_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/fineair_rareplanes_detection_only_balanced_rareplanes_test.json'},
    'dota-aircraft-y11s-detonly-s1': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_detection_only_seed1_test.json'},
    'dota-aircraft-y11s-detonly-s2': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_detection_only_seed2_test.json'},
    'aircraft-dota-aircraft-y11s-strict-det-s1': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed1_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed1_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed1_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-strict-det-s2': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed2_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed2_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_strict_detached_seed2_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-s1': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed1_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed1_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed1_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dota-adapter-s2': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed2_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed2_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dota_adapter_seed2_dota_aircraft_test.json'},
    'dota-aircraft-gsd030-y11s-detonly-s0': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_gsd030_detection_only_seed0_test.json'},
    'aircraft-dota-aircraft-gsd030-y11s-dota-adapter-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_gsd030_dota_adapter_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_gsd030_dota_adapter_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_gsd030_dota_adapter_dota_aircraft_test.json'},
    'dota-aircraft-fda-y11s-detonly-s0': {
        'dota_aircraft': ROOT / 'evaluations/dota_aircraft_fda_detection_only_seed0_test.json'},
    'aircraft-dota-aircraft-fda-y11s-dota-adapter-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_fda_dota_adapter_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_fda_dota_adapter_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_fda_dota_adapter_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-early-backbone-routes-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-early-backbone-routes-s1': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_seed1_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_seed1_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_early_backbone_routes_seed1_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-conditional-heads-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-conditional-heads-s1': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_seed1_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_seed1_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_conditional_heads_seed1_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dynamic-adapter-s0': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_dota_aircraft_test.json'},
    'aircraft-dota-aircraft-y11s-dynamic-adapter-s1': {
        'fineair': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_seed1_fineair_test.json',
        'rareplanes': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_seed1_rareplanes_test.json',
        'dota_aircraft': ROOT / 'evaluations/aircraft_dota_aircraft_dynamic_adapter_seed1_dota_aircraft_test.json'},
}

FIELDS = (
    'dataset_scope', 'variant', 'sharing_policy', 'status', 'wandb_run',
    'best_epoch', 'early_stopping_metric', 'val_map50', 'val_map50_95',
    'val_fineair_map50', 'val_fineair_map50_95',
    'val_rareplanes_map50', 'val_rareplanes_map50_95',
    'val_dota_ships_map50', 'val_dota_ships_map50_95',
    'val_dota_aircraft_map50', 'val_dota_aircraft_map50_95',
    'val_dota_aircraft_mask_iou',
    'val_mean_map50', 'val_mean_map50_95',
    'test_map50', 'test_map50_95',
    'test_fineair_map50', 'test_fineair_map50_95',
    'test_rareplanes_map50', 'test_rareplanes_map50_95', 'test_mean_map50_95',
    'test_dota_ships_map50', 'test_dota_ships_map50_95',
    'test_dota_aircraft_map50', 'test_dota_aircraft_map50_95',
    'test_fine_accuracy', 'test_very_fine_accuracy', 'test_role_accuracy',
    'test_attribute_macro_accuracy', 'test_fuselage_length_mae',
    'test_wingspan_mae', 'checkpoint', 'updated_utc',
)


def wandb_states():
    try:
        import wandb
        api = wandb.Api(timeout=15)
        states = {}
        for _, _, _, run_id, _ in EXPERIMENTS:
            try:
                states[run_id] = api.run(f'{PROJECT}/{run_id}').state
            except Exception as error:
                print(f'W&B status unavailable for {run_id}: {error}')
        return states
    except Exception as error:
        print(f'W&B status unavailable: {error}')
        return {}


def test_values(run_id):
    paths = TEST_RESULTS.get(run_id, {})
    results = {
        dataset: json.loads(path.read_text())
        for dataset, path in paths.items() if path.exists()
    }
    if not results:
        return {}
    values = {}
    for dataset, result in results.items():
        values[f'test_{dataset}_map50'] = result['detection'].get('map50')
        values[f'test_{dataset}_map50_95'] = result['detection'].get('map50_95')
    if len(results) == 1:
        result = next(iter(results.values()))
        values['test_map50'] = result['detection'].get('map50')
        values['test_map50_95'] = result['detection'].get('map50_95')
    if len(results) >= 2:
        values['test_mean_map50_95'] = sum(
            result['detection']['map50_95'] for result in results.values()) / len(results)

    classification = {
        task: metrics for result in results.values()
        for task, metrics in result.get('classification', {}).items()
        if metrics.get('total')
    }
    values.update({
        'test_fine_accuracy': classification.get('fine-class', {}).get('accuracy'),
        'test_very_fine_accuracy': classification.get('very-fine-class', {}).get('accuracy'),
        'test_role_accuracy': classification.get('role', {}).get('accuracy'),
    })
    attribute_accuracies = [
        metrics['accuracy'] for task, metrics in classification.items()
        if task.startswith('attributes_') and metrics.get('accuracy') is not None
    ]
    if attribute_accuracies:
        values['test_attribute_macro_accuracy'] = (
            sum(attribute_accuracies) / len(attribute_accuracies))
    regression = {
        task: metrics for result in results.values()
        for task, metrics in result.get('regression', {}).items()
        if metrics.get('total')
    }
    values['test_fuselage_length_mae'] = regression.get(
        'attributes_fuselage_length', {}).get('original_scale_mae')
    values['test_wingspan_mae'] = regression.get(
        'attributes_wings_wing-span', {}).get('original_scale_mae')
    return values


def checkpoint_values(folder):
    path = folder / 'best.pt'
    if not path.exists():
        return {'checkpoint': str(path)}
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metrics = checkpoint.get('validation_metrics', {})
    values = {
        'checkpoint': str(path),
        'best_epoch': int(checkpoint['epoch']) + 1,
        'early_stopping_metric': checkpoint.get('early_stopping_metric'),
    }
    for metric in (
        'map50', 'map50_95', 'fineair_map50', 'fineair_map50_95',
        'rareplanes_map50', 'rareplanes_map50_95', 'mean_map50', 'mean_map50_95',
        'dota_ships_map50', 'dota_ships_map50_95',
        'dota_aircraft_map50', 'dota_aircraft_map50_95',
        'dota_aircraft_mask_iou',
        'mask_iou',
    ):
        values[f'val_{metric}'] = metrics.get(metric)
    return values


def populate_dataset_columns(row):
    """Mirror single-dataset metrics into the common comparison columns."""
    dataset = row['dataset_scope'].lower()
    if dataset not in {'fineair', 'rareplanes', 'dota-aircraft'}:
        return
    dataset = dataset.replace('-', '_')
    for split in ('val', 'test'):
        for metric in ('map50', 'map50_95'):
            value = row.get(f'{split}_{metric}')
            if value is not None:
                row[f'{split}_{dataset}_{metric}'] = value
    if dataset == 'dota_aircraft' and row.get('val_mask_iou') is not None:
        row['val_dota_aircraft_mask_iou'] = row['val_mask_iou']


def independent_reference_rows(rows, updated):
    """Create macro references from separately trained single-dataset models.

    These are deliberately separate rows: their two checkpoints were selected
    independently and must not be presented as one jointly trained model.
    """
    by_key = {(row['dataset_scope'], row['variant']): row for row in rows}
    references = []
    for variant in ('joint', 'detached', 'detection_only'):
        fineair = by_key.get(('FineAir', variant))
        rareplanes = by_key.get(('RarePlanes', variant))
        if not fineair or not rareplanes:
            continue
        reference = {
            'dataset_scope': 'FineAir+RarePlanes',
            'variant': f'independent_{variant}',
            'sharing_policy': 'two independently trained models; macro-average reference',
            'status': ('finished' if fineair.get('status') == rareplanes.get('status') == 'finished'
                       else f"FineAir={fineair.get('status')};RarePlanes={rareplanes.get('status')}"),
            'wandb_run': '',
            'early_stopping_metric': 'independent per-dataset selection',
            'checkpoint': f"{fineair.get('checkpoint')};{rareplanes.get('checkpoint')}",
            'updated_utc': updated,
        }
        for split in ('val', 'test'):
            for metric in ('map50', 'map50_95'):
                fineair_value = fineair.get(f'{split}_fineair_{metric}')
                rareplanes_value = rareplanes.get(f'{split}_rareplanes_{metric}')
                reference[f'{split}_fineair_{metric}'] = fineair_value
                reference[f'{split}_rareplanes_{metric}'] = rareplanes_value
                if fineair_value is not None and rareplanes_value is not None:
                    reference[f'{split}_mean_{metric}'] = (
                        fineair_value + rareplanes_value) / 2
        references.append(reference)
    return references


def main():
    states = wandb_states()
    updated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    for scope, variant, sharing, run_id, folder in EXPERIMENTS:
        row = {
            'dataset_scope': scope,
            'variant': variant,
            'sharing_policy': sharing,
            'status': states.get(run_id, 'unknown'),
            'wandb_run': f'https://wandb.ai/{PROJECT}/runs/{run_id}',
            'updated_utc': updated,
        }
        row.update(checkpoint_values(folder))
        row.update(test_values(run_id))
        populate_dataset_columns(row)
        rows.append(row)

    rows.extend(independent_reference_rows(rows, updated))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'updated {OUTPUT} with {len(rows)} experiments')


if __name__ == '__main__':
    main()
