from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Selection(BaseModel):
    target: str | None = None
    task: Literal['classification', 'regression'] = 'classification'
    features: list[str] = Field(default_factory=list)
    identifier: str | None = None


class Operation(BaseModel):
    kind: Literal['impute', 'trim', 'drop_duplicates', 'drop_column', 'clip']
    column: str | None = None
    method: Literal['median', 'mean', 'most_frequent', 'missing_category'] | None = None


class RepairRequest(BaseModel):
    operations: list[Operation] = Field(default_factory=list, max_length=200)


class TrainingConfig(BaseModel):
    dataset_id: str
    version_id: str
    target: str
    task: Literal['classification', 'regression']
    features: list[str] = Field(min_length=1, max_length=200)
    algorithm: Literal['logistic', 'tree', 'forest', 'gradient', 'linear', 'ridge'] = 'forest'
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    validation_size: float = Field(default=.2, ge=.1, le=.35)
    test_size: float = Field(default=.2, ge=.1, le=.35)
    split: Literal['random', 'group', 'chronological'] = 'random'
    group_column: str | None = None
    time_column: str | None = None
    identifier: str | None = None
    stratify: bool = True
    n_estimators: int = Field(default=100, ge=10, le=300)
    max_depth: int | None = Field(default=None, ge=1, le=30)
    class_weight: Literal['none', 'balanced'] = 'none'
    imputation: Literal['median', 'mean'] = 'median'

    @model_validator(mode='after')
    def validate_sizes(self):
        if self.validation_size + self.test_size > .65:
            raise ValueError('Reserve at least 35% of data for training.')
        if self.task == 'classification' and self.algorithm in ('linear', 'ridge'):
            raise ValueError('Linear and ridge algorithms require regression.')
        if self.task == 'regression' and self.algorithm == 'logistic':
            raise ValueError('Logistic regression requires classification.')
        if self.class_weight == 'balanced' and (self.task != 'classification' or self.algorithm == 'gradient'):
            raise ValueError('Balanced class weights are supported for classification logistic, tree and forest models.')
        return self
