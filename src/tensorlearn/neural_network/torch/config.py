from typing import Optional, Union, List, Callable
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from tensorlearn.neural_network.torch import utils


class LRTLinearConfig(BaseModel):
    mode: Optional[Union[str,None]]
    decomp_format: Optional[str] = None
    dim_list: Optional[List[int]] = None
    rank_list: Optional[List[int]] = None
    bias: Optional[bool] = None
    weight_transpose: Optional[bool] = None
    in_order: Optional[int] = None
    out_order: Optional[int]=None
    shape_search_method: Optional[str]=None
    error: Optional[float]=None
    get_shape:Optional[Callable]=None
    get_factors:Optional[Callable]=None

    @field_validator('mode')
    def check_mode(cls,value):
        if value is not None and value not in {'default', 'teacher'}:
            raise ValueError("mode must be 'default' or 'teacher' or None")
        return value
    
    @model_validator(mode='before')
    def check_params_before(cls, values):
        mode = values.get('mode')
        if mode == 'default':
            required_fields = ['decomp_format', 'dim_list', 'rank_list', 'in_order', 'bias']
        elif mode == 'teacher':
            required_fields = ['decomp_format','in_order', 'out_order', 'weight_transpose', 'shape_search_method', 'error']
        else:
            required_fields = []

        for field in required_fields:
            if values.get(field) is None:
                raise ValueError(f"{field} is required for {mode} mode")
        return values

    def __init__(self,**data):
        super().__init__(**data)
        if self.mode=='teacher':
            if self.shape_search_method=='balanced':
                self.get_shape=utils.get_tensorized_layer_balanced_shape
                if self.decomp_format=='tt':
                    self.get_factors=utils.get_tt_factors #for GA this get_factor will not work as it does not address padding
                elif self.decomp_format=='tucker':
                    self.get_factors=utils.get_tucker_factors
                else:
                    raise ValueError("decomp format is not recognized")
            else: 
                raise ValueError("shape search method is not recognized")
        
        

        



