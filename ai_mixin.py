import numpy as np
import requests
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

def _check_transformers():
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
        return True
    except ImportError:
        return False

class InferenceWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self._func = func
        self._args = args

    def run(self):
        try:
            self.finished.emit(self._func(*self._args))
        except Exception as e:
            self.error.emit(str(e))


class AIMixin:
    def _append_chat(self, html: str):
        if not hasattr(self, '_chat_lines'):
            self._chat_lines = []
        self._chat_lines.append(html)
        self._refresh_chat()

    def _refresh_chat(self):
        self.ai_chat_history.setHtml(
            "<html><body style='font-family:sans-serif;'>" +
            "".join(self._chat_lines) +
            "</body></html>"
        )
        sb = self.ai_chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def load_transformers_model(self):
        if not _check_transformers():
            QMessageBox.warning(self, '警告', '未安装必要库，请运行:\npip install transformers torch')
            return
        model_path = self.ai_transformers_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, '警告', '请输入模型路径或 HuggingFace 模型名称\n例如: Qwen/Qwen3-0.6B')
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._append_chat(
                f"<p style='color:#5a6c7d;'>⏳ 正在加载模型: {model_path}，首次下载可能需要几分钟...</p>")
            QApplication.processEvents()
            self.local_llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            self.local_llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda")
            self.local_llm_model.eval()
            torch.cuda.empty_cache()
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info = f"GPU: {gpu_name} ({vram:.1f} GB VRAM)"
            self._append_chat(
                f"<p style='color:green;'>✓ 模型 <b>{model_path}</b> 加载成功！运行设备: {device_info}</p>")
            QMessageBox.information(self, '成功', f'模型已加载到 {device_info}\n\n提示: Qwen3 模型支持深度思考模式，本程序默认关闭以加快响应速度。')
        except Exception as e:
            self._append_chat(f"<p style='color:red;'>❌ 模型加载失败: {str(e)}</p>")
            QMessageBox.critical(self, '错误', f'模型加载失败:\n{str(e)}\n\n请确认模型名称正确，例如: Qwen/Qwen3-0.6B')

    def send_ai_message(self):
        user_message = self.ai_input.text().strip()
        if not user_message:
            return
        if getattr(self, '_inference_worker', None) and self._inference_worker.isRunning():
            return
        self._append_chat(f"<p style='color:#5a6c7d;'><b>👤 您:</b> {user_message}</p>")
        self.ai_input.clear()
        self.ai_input.setEnabled(False)
        self._thinking_dot_count = 0
        self._thinking_idx = len(getattr(self, '_chat_lines', []))
        self._append_chat("<p style='color:#888;'><b>🤔 AI助手:</b> 思考中</p>")
        context = self.build_ai_context()
        system_prompt = (
            "你是一个专业的数据分析与机器学习助手，帮助用户解读 Excel 数据分析和机器学习模型结果。\n\n"
            "当前程序已完成的分析信息如下：\n"
            f"{context}\n\n"
            "请根据用户问题，结合以上信息给出专业、清晰、有针对性的回答。"
            "如涉及模型性能，请给出改进建议；如涉及数据，请结合统计信息回答。"
        )
        func = self.call_transformers_model if self.radio_transformers.isChecked() else self.call_api_model
        self._inference_worker = InferenceWorker(func, system_prompt, user_message)
        self._inference_worker.finished.connect(self._on_inference_done)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.start()
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_thinking)
        self._dot_timer.start(400)

    def _animate_thinking(self):
        self._thinking_dot_count = (self._thinking_dot_count + 1) % 4
        dots = "·" * self._thinking_dot_count
        lines = getattr(self, '_chat_lines', [])
        if self._thinking_idx < len(lines):
            lines[self._thinking_idx] = f"<p style='color:#888;'><b>🤔 AI助手:</b> 思考中{dots}</p>"
            self._refresh_chat()

    def _stop_thinking(self):
        if hasattr(self, '_dot_timer') and self._dot_timer.isActive():
            self._dot_timer.stop()
        self.ai_input.setEnabled(True)

    def _on_inference_done(self, response: str):
        self._stop_thinking()
        lines = getattr(self, '_chat_lines', [])
        if self._thinking_idx < len(lines):
            lines[self._thinking_idx] = f"<p style='color:#2c5282;'><b>🤖 AI助手:</b> {response}</p>"
            self._refresh_chat()

    def _on_inference_error(self, error_msg: str):
        self._stop_thinking()
        lines = getattr(self, '_chat_lines', [])
        if self._thinking_idx < len(lines):
            lines[self._thinking_idx] = f"<p style='color:red;'><b>❌ 错误:</b> {error_msg}</p>"
            self._refresh_chat()

    def call_transformers_model(self, system_prompt, user_message):
        if self.local_llm_model is None or self.local_llm_tokenizer is None:
            raise Exception("请先点击「加载模型」按钮加载本地模型")
        import torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        tokenizer = self.local_llm_tokenizer
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            text = f"{system_prompt}\n\n用户: {user_message}\n\nAI助手:"

        device = next(self.local_llm_model.parameters()).device
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            generated_ids = self.local_llm_model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not response:
            full = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response = full.split("AI助手:")[-1].strip()
        return response

    def build_ai_context(self):
        context = []

        if self.df is not None:
            context.append("【数据集信息】")
            context.append(f"- 行数: {len(self.df)}, 列数: {len(self.df.columns)}")
            context.append(f"- 列名: {', '.join(self.df.columns.tolist())}")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            cat_cols     = self.df.select_dtypes(include=['object']).columns
            if len(numeric_cols) > 0:
                context.append(f"- 数值型列 ({len(numeric_cols)}个): {', '.join(numeric_cols.tolist())}")
            if len(cat_cols) > 0:
                context.append(f"- 分类型列 ({len(cat_cols)}个): {', '.join(cat_cols.tolist())}")
            missing = self.df.isnull().sum().sum()
            if missing > 0:
                context.append(f"- 缺失值总数: {missing}")
            if len(numeric_cols) > 0:
                desc = self.df[numeric_cols[:8]].describe().round(4)
                context.append("- 数值统计摘要 (均值/标准差/最小/最大):")
                for col in numeric_cols[:8]:
                    s = desc[col]
                    context.append(f"  · {col}: 均值={s['mean']:.4f}, 标准差={s['std']:.4f}, 最小={s['min']:.4f}, 最大={s['max']:.4f}")
            if len(cat_cols) > 0:
                context.append("- 分类列主要取值:")
                for col in cat_cols[:5]:
                    top = self.df[col].value_counts().head(5).to_dict()
                    context.append(f"  · {col}: {top}")
            sample_str = self.df.head(3).to_string(index=False)
            context.append(f"- 数据前3行样例:\n{sample_str}")

        if self.trained_model is not None:
            model_name = type(self.trained_model).__name__
            context.append(f"\n【已训练的单一模型】")
            context.append(f"- 模型类型: {model_name}")
            if hasattr(self.trained_model, 'get_params'):
                params = {k: v for k, v in self.trained_model.get_params().items()
                          if v is not None and k in ('n_estimators','max_depth','C','learning_rate','num_leaves')}
                if params:
                    context.append(f"- 主要参数: {params}")
            if hasattr(self.trained_model, 'feature_importances_') and self.df is not None:
                importances = self.trained_model.feature_importances_
                feat_names  = [cb.text() for i in range(self.feature_checkboxes_layout.count())
                               for cb in [self.feature_checkboxes_layout.itemAt(i).widget()]
                               if cb and cb.isChecked()]
                if len(feat_names) == len(importances):
                    top3 = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:3]
                    context.append(f"- Top3重要特征: {', '.join(f'{n}({v:.3f})' for n,v in top3)}")

        if self.last_predictions is not None and self.last_actual is not None:
            actual    = np.asarray(self.last_actual)
            predicted = np.asarray(self.last_predictions)
            context.append("\n【测试集预测指标】")
            unique_vals = np.unique(actual)
            if len(unique_vals) <= 20 and np.all(actual == actual.astype(int)):
                acc = accuracy_score(actual.astype(int), predicted.astype(int))
                context.append(f"- 准确率 (Accuracy): {acc:.4f}")
                if acc < 0.6:
                    context.append("- ⚠️ 准确率偏低，模型需要改进")
                elif acc > 0.9:
                    context.append("- ✓ 准确率较高，但需警惕过拟合")
            else:
                r2  = r2_score(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                mae  = np.mean(np.abs(actual - predicted))
                context.append(f"- R² 决定系数: {r2:.4f}")
                context.append(f"- RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                if r2 < 0.5:
                    context.append("- ⚠️ R²较低，模型拟合效果不理想")
                elif r2 > 0.8:
                    context.append("- ✓ 模型拟合效果良好")

        if self.last_train_predictions is not None and self.last_train_actual is not None:
            train_actual = np.asarray(self.last_train_actual)
            train_pred   = np.asarray(self.last_train_predictions)
            unique_vals  = np.unique(train_actual)
            if len(unique_vals) <= 20 and np.all(train_actual == train_actual.astype(int)):
                train_acc  = accuracy_score(train_actual.astype(int), train_pred.astype(int))
                test_acc   = accuracy_score(np.asarray(self.last_actual).astype(int),
                                            np.asarray(self.last_predictions).astype(int))
                diff = train_acc - test_acc
                context.append(f"- 训练集准确率: {train_acc:.4f}，差异: {diff:+.4f}")
                if diff > 0.15:
                    context.append("- ⚠️ 过拟合严重 (训练/测试差异 > 15%)")
            else:
                train_r2 = r2_score(train_actual, train_pred)
                test_r2  = r2_score(np.asarray(self.last_actual), np.asarray(self.last_predictions))
                context.append(f"- 训练集R²: {train_r2:.4f}，差异: {train_r2-test_r2:+.4f}")
                if train_r2 - test_r2 > 0.2:
                    context.append("- ⚠️ 过拟合严重 (R²差异 > 0.2)")

        if self.trained_ensemble_model is not None and len(self.ensemble_models) > 0:
            context.append("\n【集成模型配置】")
            context.append(f"- 模型数量: {len(self.ensemble_models)}")
            for mi in self.ensemble_models:
                context.append(f"  · {mi['name']} (权重: {mi['weight']})")

        return "\n".join(context) if context else "暂无分析数据，请先加载数据并训练模型"

    def call_api_model(self, system_prompt, user_message):
        api_url    = self.ai_api_url.text().strip()
        api_key    = self.ai_api_key.text().strip()
        model_name = self.ai_model_name.text().strip()
        if not api_url:
            raise Exception("请填写 API 地址\n推荐: https://api.siliconflow.cn/v1/chat/completions")
        if not api_key:
            raise Exception("请填写 API Key\n前往 siliconflow.cn 免费注册获取")
        if not model_name:
            raise Exception("请填写模型名称\n推荐: Qwen/Qwen3-8B")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        if hasattr(self, 'ai_think_mode') and not self.ai_think_mode:
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

    def clear_ai_chat(self):
        self._chat_lines = []
        self.ai_chat_history.clear()
