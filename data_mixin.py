import os
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QCheckBox, QApplication
from PyQt5.QtCore import Qt

class DataMixin:
    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                self.statusBar().showMessage('正在加载文件...')
                QApplication.processEvents()
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
                elif file_path.endswith('.xlsx'):
                    self.df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    self.df = pd.read_excel(file_path, engine='xlrd')

                self.filtered_df = self.df.copy()
                self.combo_columns.clear()
                self.combo_columns.addItem('全部列')
                self.combo_columns.addItems(self.df.columns.tolist())
                self.update_ml_feature_list()
                self.update_ensemble_feature_list()
                self.ml_target_combo.clear()
                self.ml_target_combo.addItems(self.df.columns.tolist())
                self.ensemble_target_combo.clear()
                self.ensemble_target_combo.addItems(self.df.columns.tolist())
                self.viz_x_combo.add_items(self.df.columns.tolist())
                self.viz_y_combo.add_items(self.df.columns.tolist())
                self.display_data(self.df)
                self.btn_export.setEnabled(True)
                self.btn_refresh.setEnabled(True)

                file_size = os.path.getsize(file_path) / (1024 * 1024)
                self.statusBar().showMessage(
                    f'✓ 已加载: {os.path.basename(file_path)} ({file_size:.2f} MB) | '
                    f'{len(self.df)} 行 × {len(self.df.columns)} 列'
                )
                if file_size > 10:
                    QMessageBox.information(
                        self, '性能优化提示',
                        f'文件较大 ({file_size:.1f} MB)\n\n'
                        '💡 加速建议:\n'
                        '1. 将Excel转换为CSV格式可大幅提升加载速度 (5-10倍)\n'
                        '2. 如果只需要部分数据，可在Excel中先筛选后再导入\n'
                        '3. 考虑使用数据库存储大型数据集\n\n'
                        '转换方法: 在Excel中点击"另存为" → 选择CSV格式'
                    )
            except Exception as e:
                QMessageBox.critical(self, '错误', f'无法读取文件:\n{str(e)}')
                self.statusBar().showMessage('加载失败')

    def update_ml_feature_list(self):
        for i in reversed(range(self.feature_checkboxes_layout.count())):
            self.feature_checkboxes_layout.itemAt(i).widget().setParent(None)
        for col in self.df.columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(True)
            self.feature_checkboxes_layout.addWidget(checkbox)

    def update_ensemble_feature_list(self):
        for i in reversed(range(self.ensemble_feature_checkboxes_layout.count())):
            self.ensemble_feature_checkboxes_layout.itemAt(i).widget().setParent(None)
        for col in self.df.columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(True)
            self.ensemble_feature_checkboxes_layout.addWidget(checkbox)

    def display_data(self, df):
        if df is not None and not df.empty:
            import pandas as pd
            self.table.setSortingEnabled(False)
            self.table.setRowCount(len(df))
            self.table.setColumnCount(len(df.columns))
            self.table.setHorizontalHeaderLabels(df.columns.tolist())
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    value = df.iloc[i, j]
                    display_value = '' if pd.isna(value) else str(value)
                    item = QTableWidgetItem(display_value)
                    item.setTextAlignment(Qt.AlignCenter)
                    try:
                        numeric_value = float(value)
                        item.setData(Qt.UserRole, numeric_value)
                    except:
                        item.setData(Qt.UserRole, display_value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(i, j, item)
            self.table.resizeColumnsToContents()
            self.table.setSortingEnabled(True)
            if df is self.filtered_df and len(self.filtered_df) < len(self.df):
                self.statusBar().showMessage(
                    f'显示 {len(df)} 行 (共 {len(self.df)} 行) × {len(df.columns)} 列')
            else:
                self.statusBar().showMessage(f'共 {len(df)} 行 × {len(df.columns)} 列')
        else:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.statusBar().showMessage('无数据')

    def filter_data(self):
        if self.df is None:
            return
        search_text = self.search_input.text().strip()
        selected_column = self.combo_columns.currentText()
        if not search_text:
            self.filtered_df = self.df.copy()
        else:
            if selected_column == '全部列':
                mask = self.df.astype(str).apply(
                    lambda x: x.str.contains(search_text, case=False, na=False)
                ).any(axis=1)
            else:
                mask = self.df[selected_column].astype(str).str.contains(
                    search_text, case=False, na=False)
            self.filtered_df = self.df[mask]
        self.display_data(self.filtered_df)

    def clear_filter(self):
        self.search_input.clear()
        self.combo_columns.setCurrentIndex(0)
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self.display_data(self.df)

    def refresh_data(self):
        if self.df is not None:
            self.clear_filter()
            self.display_data(self.df)
            self.statusBar().showMessage('数据已刷新')

    def export_csv(self):
        if self.df is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存CSV文件", "", "CSV Files (*.csv);;All Files (*)")
            if file_path:
                try:
                    export_df = self.filtered_df if self.filtered_df is not None else self.df
                    export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    QMessageBox.information(
                        self, '成功',
                        f'文件已成功导出到:\n{file_path}\n\n共导出 {len(export_df)} 行数据')
                    self.statusBar().showMessage(f'✓ 已导出到: {file_path}')
                except Exception as e:
                    QMessageBox.critical(self, '错误', f'导出失败:\n{str(e)}')
                    self.statusBar().showMessage('导出失败')
        else:
            QMessageBox.warning(self, '警告', '请先加载Excel文件')