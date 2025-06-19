CODE_HOME=/Work21/2023/wangtianrui/codes/VocalStory
cd ${CODE_HOME}
# CKPT=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/celsds/logs/cosyvoice2_tp/ls_aishell12_dt1.5k_giga23k_cv_nonext_zip_nar_conv_balloss_smalllr/checkpoint.best_loss_0.5242.pt

CUDA_VISIBLE_DEVICES=5 \
COSYVOICE2HOME=model_cache/CosyVoice2-0.5B \
PYTHONPATH=${CODE_HOME}:${CODE_HOME}/wescon/Matcha-TTS \
/Work21/2023/wangtianrui/miniconda3/envs/vocalstory/bin/python -u \
${CODE_HOME}/wescon/narration_api.py 2>&1 | tee /Work21/2023/wangtianrui/codes/VocalStory/outputs/run_narration.out &



# kill -9 $(ps aux | grep "wescon" | grep -v grep | awk '{print $2}')