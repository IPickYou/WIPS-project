class DataProcessor:
    @staticmethod
    def validate_dataframe(df, required_cols=None):
        if df is None or len(df) == 0:
            raise ValueError("데이터가 비어 있습니다.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(missing_cols)

        return True

    @staticmethod
    def get_available_columns(df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = ["사용자태그", "WINTELIPS KEY"]

        return [col for col in df.columns if col not in exclude_cols]

    @staticmethod
    def create_results_summary(results_df, label_col='예측_라벨'):

        if results_df is None or len(results_df) == 0:
            return None

        summary = {
            'total_count': len(results_df),
            'label_distribution': results_df[label_col].value_counts().to_dict(),
            'unique_labels': results_df[label_col].nunique()
        }

        if '신뢰도' in results_df.columns:
            summary['confidence_stats'] = {
                'mean': results_df['신뢰도'].mean(),
                'min': results_df['신뢰도'].min(),
                'max': results_df['신뢰도'].max(),
                'std': results_df['신뢰도'].std()
            }

        return summary