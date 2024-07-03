

###############################################################################
###############################################################################
###############################################################################

import numpy as np



###############################################################################
###############################################################################
###############################################################################

class ga():
    

    #############################################################
    def __init__(self, function, dimension, \
                 original_input_shape,\
                 lower_bound=2,\
                 initial_shape=True,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     #convergence_curve=True,\
                         #progress_bar=True
                         ):



        self.__name__=ga
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        self.dim=int(dimension)
        self.original_input_shape=original_input_shape
    

        temp_tensor_size=1
        
        for s in original_input_shape:
            temp_tensor_size*=s

        self.tensor_size=temp_tensor_size
        input_shape_array=np.array(original_input_shape)
        ones=np.where(input_shape_array==1)
        
        removed_ones=np.delete(input_shape_array,ones)

        if len(removed_ones)>=3 and initial_shape and len(removed_ones)<=self.dim:
            self.init_func=self.given_shape
            
        else:
            self.init_func=self.random_init
            
        
        
        self.params=self.dim-1

        self.param_list=np.arange(0,self.dim)

        self.lower_bound=lower_bound
        self.lb=[lower_bound]*self.dim
        if lower_bound==1:

            #lb_list=np.random.permutation(self.param_list)
            
            for i in range(0,3):
                self.lb[i]=2
        
        ############################################################# 
        #convergence_curve
        #if convergence_curve==True:
        #    self.convergence_curve=True
        #else:
        #    self.convergence_curve=False
        ############################################################# 
        #progress_bar
        #if progress_bar==True:
        #    self.progress_bar=True
        #else:
        #    self.progress_bar=False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters
        
        self.param=algorithm_parameters
        
        self.pop_s=int(self.param['population_size'])
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=self.param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])

        
        ############################################################# 
    def run(self):
        
        
        ############################################################# 
        # Initial Population
        


        pop,solo,var,obj=self.init_func(self.original_input_shape)
        

        

            
        
        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        ##############################################################   
        
        t=1
        counter=0
        while t<=self.iterate:
            
            #if self.progress_bar==True:
                #self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

                
            
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])
    
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
                
                
            parallel_list=[]
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                
                
                
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mut(ch2)  
                
                parallel_list.append(ch1)
                parallel_list.append(ch2)
                solo[: self.dim]=ch1.copy()   
                obj=self.f(ch1)
                solo[self.dim]=obj
                pop[k]=solo.copy()                
                solo[: self.dim]=ch2.copy()                
                obj=self.f(ch2)               
                solo[self.dim]=obj
                pop[k+1]=solo.copy()
            
            #print(pop)

                
                
        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    #if self.progress_bar==True:
                        #self.progress(t,self.iterate,status="GA is running...")
                    #time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
                
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report
        
        self.report.append(pop[0,self.dim])
        
        
 
        
        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        #if self.progress_bar==True:
        #    show=' '*100
        #    sys.stdout.write('\r%s' % (show))
        #sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        #sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        #sys.stdout.flush() 
        #re=np.array(self.report)

        #if self.convergence_curve==True:
        #    plt.plot(re)
        #    plt.xlabel('Iteration')
        #    plt.ylabel('Objective function')
        #    plt.title('Genetic Algorithm')
        #    plt.show()
        
        #if self.stop_mniwi==True:
        #    sys.stdout.write('\nWarning: GA is terminated due to the'+\
        #                     ' maximum number of iterations without improvement was met!')
       
        
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        

        if c_type=='one_point':
            ran=np.random.randint(0,self.params)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
  
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.params)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.params):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                    
        ofs1=self.modify(ofs1)
        ofs2=self.modify(ofs2)
                   
        return np.array([ofs1,ofs2])
###############################################################################  

    def mut(self,x):
        
        
        
        param_list=np.random.permutation(self.param_list)
        
        i=np.random.randint(0,self.params)
            
        cardinality=1
            
        for j in range(0,self.params):
                cardinality=cardinality*x[param_list[j]]
            
        cardinality=cardinality/x[param_list[i]]
            
            
            
        upper_bound=int(np.ceil(self.tensor_size/(cardinality*self.lower_bound)))
        
        x[param_list[i]]=np.random.randint(self.lower_bound,upper_bound+1) 
        cardinality=cardinality*x[param_list[i]]
            
        x[param_list[-1]]=int(np.ceil(self.tensor_size/(cardinality)))
        

        x=self.one_double_check(x)
  
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
############################################################################### 
        
    def modify(self,x):
        
        
        param_list=np.random.permutation(self.param_list)  
        
        #self.lb=np.random.permutation(self.lb)

        lb_mult=np.prod(self.lb)
        lb_mult_partial=lb_mult/self.lb[0]
            
        upper_b_0=int(np.ceil(self.tensor_size/(lb_mult_partial)))

        while x[param_list[0]]>upper_b_0:
            x[param_list[0]]-=1
        mult_pre=1
        for i in range(1,self.params):
            
            
            mult_pre=mult_pre*x[param_list[i-1]]
            lb_mult_partial=lb_mult_partial/self.lb[i]
            upper_b=int(np.ceil(self.tensor_size/(mult_pre*(lb_mult_partial))))
            
            
            while x[param_list[i]]>upper_b:
                x[param_list[i]]-=1
                
                
        mult_pre=mult_pre*x[param_list[-2]]
        
        while x[param_list[-1]]>int(np.ceil(self.tensor_size/(mult_pre))):
            #print(x[-1])
            #print(int(np.ceil(self.tensor_size/(mult_pre))))
            #print(mult_pre)
            x[param_list[-1]]-=1
            
            
        #print(x)
        #for i in range(0,self.dim):
           # if x[i]<self.lb[i]:
            #    print(x)
            #    print(self.lb)
             #   raise 'error in lower bound'


          
        tes=np.where(x==1)
        test=np.delete(x,tes)    
        while len(test)<3:
            
            rand=np.random.randint(0,len(tes))
            x[tes[rand]]+=1
            tes=np.where(x==1)
            test=np.delete(x,tes) 


        
        
        
        cardinality=1
        for i in range(self.dim):
            cardinality=cardinality*x[param_list[i]]
          
        cardinality_partial=cardinality/x[param_list[-1]]
        cardinality_invest=cardinality_partial*(x[param_list[-1]]-1)
        
        
        while cardinality_invest>self.tensor_size:
            if x[param_list[-1]]>self.lb[param_list[-1]]:
                x[param_list[-1]]-=1
                cardinality=cardinality_partial*x[param_list[-1]]
                cardinality_invest=cardinality_partial*(x[param_list[-1]]-1)
            else:
                
                condition=True
                while condition:
                    rand=np.random.randint(0,self.params)
                    if x[param_list[rand]]>2:
                        cardinality=cardinality/x[param_list[rand]]
                        cardinality_partial=cardinality_partial/x[param_list[rand]]
                        cardinality_invest=cardinality_invest/x[param_list[rand]]
                        
                        
                        x[param_list[rand]]-=1
                        cardinality=cardinality*x[param_list[rand]]
                        cardinality_partial=cardinality_partial*x[param_list[rand]]
                        cardinality_invest=cardinality_invest*x[param_list[rand]]
                        
                        condition=False
        
                    
            
        
        while cardinality<self.tensor_size:
            cardinality=cardinality/x[param_list[-1]]
            x[param_list[-1]]=x[param_list[-1]]+1
            cardinality=cardinality*x[param_list[-1]]
        

        #print(x)
        return x
            
###############################################################################          
    def one_double_check(self,x):
        ones=np.where(x==1)
        test=np.delete(x,ones)
        while len(test)<3:
                
                rand=np.random.randint(0,len(ones))
                x[ones[rand]]+=1
                ones=np.where(x==1)
                test=np.delete(x,ones) 

        return x

    def evaluate(self):
        
        return self.f(self.temp)
###############################################################################    
   # def sim(self,X):
    #    self.temp=X.copy()
    #    obj=None
    #    try:
    #        obj=func_timeout(self.funtimeout,self.evaluate)
    #    except FunctionTimedOut:
    #        print("given function is not applicable")
    #    assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
    #            "func_timeout: the given function does not provide any output"
    #    return obj

###############################################################################
    #def progress(self, count, total, status=''):
     #   bar_len = 50
     #   filled_len = int(round(bar_len * count / float(total)))

     #   percents = round(100.0 * count / float(total), 1)
     #   bar = '|' * filled_len + '_' * (bar_len - filled_len)

     #   sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
     #   sys.stdout.flush()     
###############################################################################            
###############################################################################

    def random_init(self,shape):
        

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
        return pop,solo,var, obj
    



    def given_shape(self,shape):

        var=np.ones(self.dim) 
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)   

        shape_array=np.array(shape)
        if len(shape)==self.dim:
            var=shape_array
        elif len(shape)<self.dim:
            var[:len(shape)]=shape_array

        elif len(shape)>self.dim:
            ones=np.where(shape_array==1)
            
            removed_ones=np.delete(shape_array,ones)

            if len(removed_ones)<=self.dim and len(removed_ones)>=3:
                var[:len(removed_ones)]=removed_ones
            else:
                raise 'error in given_shape'
        
        
        solo[:self.dim]=var.copy()
        
        obj=self.f(var) 
        solo[self.dim]=obj
        pop[0]=solo.copy()

        for p in range(1,self.par_s):


            new_var=self.mut(var)
            for i in range(0,self.dim):
                solo[i]=new_var[i].copy()

            obj=self.f(new_var)
            solo[self.dim]=obj
            pop[p]=solo.copy()




        lb_mult=np.prod(self.lb)

        for p in range(self.par_s,self.pop_s):

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


    

        return pop,solo,var, obj
    

            
            
            
