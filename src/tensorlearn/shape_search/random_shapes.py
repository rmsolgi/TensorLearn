import numpy as np


class random_search:
    def __init__(self,function, input_tensor_size, dimension, low_bound_dim, number_of_trials):
          
        self.pop_s=number_of_trials
        self.dim=int(dimension)
        self.f=function
        self.tensor_size=input_tensor_size
        self.params=self.dim-1
        self.param_list=np.arange(0,self.dim)
        self.lower_bound=low_bound_dim
        self.lb=[low_bound_dim]*self.dim
        if low_bound_dim==1:

            #lb_list=np.random.permutation(self.param_list)
            
            for i in range(0,3):
                self.lb[i]=2


    def run(self):
        pop=self.random()

        pop = pop[pop[:,self.dim].argsort()]

        self.results=pop
        self.best_shape=pop[0,:self.dim]
        self.best_function=pop[0,self.dim]

     

    def random(self):
            

            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            solo=np.zeros(self.dim+1)
            var=np.zeros(self.dim)       
            
            lb_mult=np.prod(self.lb)
            #self.upper_b_0_0=int(np.ceil(self.tensor_size/(lb_mult)))

            for p in range(0,self.pop_s):

                #self.lb=np.random.permutation(self.lb)
                
                param_list=np.random.permutation(self.param_list)

                lb_mult_partial=lb_mult/self.lb[0]
                
                upper_b_0=int(np.ceil(self.tensor_size/(lb_mult_partial)))
            
                var[param_list[0]]=np.random.randint(self.lb[0],(upper_b_0)+1)
                
                solo[param_list[0]]=var[param_list[0]].copy()
                
                mult_pre=1


                for i in range(1,self.params):
                    
                    mult_pre=mult_pre*var[param_list[i-1]]

                    lb_mult_partial=lb_mult_partial/self.lb[i]
                    
                    upper_b=int(np.ceil(self.tensor_size/(mult_pre*(lb_mult_partial))))
                                    
                                    
                    
                    var[param_list[i]]=np.random.randint(self.lb[i],upper_b+1)  
                    solo[param_list[i]]=var[param_list[i]].copy()
                
                mult_pre=mult_pre*var[param_list[self.params-1]]
                
                var[param_list[-1]]=int(np.ceil(self.tensor_size/(mult_pre)))
                

                
                solo[param_list[-1]]=var[param_list[-1]].copy()
                

                obj=self.f(var)            
                solo[self.dim]=obj
                pop[p]=solo.copy()
            return pop
        



