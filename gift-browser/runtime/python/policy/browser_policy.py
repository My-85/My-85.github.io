"""Complete frozen GIFT9a policy in a browser: NumPy orchestration + 25 ONNX graphs.

No JAX/Flax/checkpoint/source checkout is imported. Research methods are the
source-hashed minimal AST closure in policy_manifest.json. This file runs in
Pyodide as well as normal CPython for independent archived-input validation.
"""
import sys,json,types,importlib,collections,time,ast,operator
from pathlib import Path
import numpy as np

CHECKPOINT='9a45713d3f392f7762758b27535e8057f271035880f00bbd99197efcde878bc0'

class _OpaqueArray:
    def __init__(self,token,spec):self.token=token;self.shape=tuple(spec['shape']);self.dtype=np.dtype(spec['dtype'])
    def __array__(self,*args,**kwargs):raise RuntimeError('Numerical weights must run in ONNX, never NumPy/JAX fallback: '+self.token)

class _ParameterArray(np.ndarray):
    def __new__(cls,data,dtype,token):
        obj=np.asarray(data,dtype=dtype).view(cls);obj.token=token;return obj
    def __array_finalize__(self,obj):self.token=getattr(obj,'token',None)

class _Policy:
    def __getattr__(self,key):return getattr(self.source,key)

def _numpy_set(value,index,update):
    result=np.array(value,copy=True);result[index]=update;return result

def _numpy_binary(name,a,b):
    # JAX keeps float32 when combining it with int32, while NumPy promotes that
    # pair to float64. Restore the original JAX promotion at source boundaries.
    ta=np.asarray(a) if isinstance(a,(np.ndarray,np.generic)) else None
    tb=np.asarray(b) if isinstance(b,(np.ndarray,np.generic)) else None
    typed=[v for v in (ta,tb) if v is not None]
    floating=any(v.dtype==np.float32 for v in typed)
    floating=floating or (bool(typed) and (isinstance(a,float) or isinstance(b,float)))
    if floating:a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    return getattr(operator,name)(a,b)

class _Math(ast.NodeTransformer):
    def visit_BinOp(self,node):
        self.generic_visit(node)
        op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'truediv',ast.Pow:'pow',ast.MatMult:'matmul',ast.FloorDiv:'floordiv',ast.Mod:'mod'}.get(type(node.op))
        if op:return ast.copy_location(ast.Call(ast.Name('_numpy_binary',ast.Load()),[ast.Constant(op),node.left,node.right],[]),node)
        return node

def _compile(source,name):
    tree=ast.parse('from __future__ import annotations\n'+source)
    tree=_Math().visit(tree);ast.fix_missing_locations(tree)
    return compile(tree,name,'exec')

def _install_numpy_shim():
    # Only source-compatible array/helper operations exist. There is deliberately
    # no network, jit compiler, CUDA backend, or neural-evaluation fallback.
    jax=types.ModuleType('jax');jnp=types.ModuleType('jax.numpy')
    jnp.__dict__.update(np.__dict__);jnp.__name__='jax.numpy'
    def asarray(value,dtype=None):
        a=np.asarray(value,dtype=dtype)
        if dtype is None:
            if a.dtype==np.float64:a=a.astype(np.float32)
            elif a.dtype==np.int64:a=a.astype(np.int32)
        return a
    def arange(*args,**kwargs):
        if 'dtype' not in kwargs:kwargs['dtype']=np.float32 if any(isinstance(v,float) for v in args) else np.int32
        return np.arange(*args,**kwargs)
    jnp.asarray=asarray;jnp.array=asarray;jnp.arange=arange
    jnp.zeros=lambda shape,dtype=np.float32:np.zeros(shape,dtype=dtype)
    jnp.ones=lambda shape,dtype=np.float32:np.ones(shape,dtype=dtype)
    jnp.full=lambda shape,fill_value,dtype=np.float32:np.full(shape,fill_value,dtype=dtype)
    jax.numpy=jnp;jax.Array=np.ndarray;jax.device_get=lambda x:x
    jax.lax=types.SimpleNamespace(bitcast_convert_type=lambda x,dtype:np.asarray(x).view(dtype))
    # Original constructors are not executed; jit means only helper identity.
    jax.jit=lambda fn,*a,**k:fn
    sys.modules['jax']=jax;sys.modules['jax.numpy']=jnp
    return jax,jnp


def _signature(v):
    if isinstance(v,dict):return ('dict',tuple((str(k),_signature(v[k])) for k in sorted(v)))
    if isinstance(v,(tuple,list)):return (type(v).__name__,tuple(_signature(x) for x in v))
    if isinstance(v,(_OpaqueArray,_ParameterArray)):return ('array',v.token)
    return ('scalar',v)


def _flatten(value):
    if isinstance(value,dict):
        for key in sorted(value):yield from _flatten(value[key])
    elif isinstance(value,(tuple,list)):
        for item in value:yield from _flatten(item)
    else:yield value


def _unflatten(tree,values):
    if tree['kind']=='leaf':return next(values)
    items=[_unflatten(child,values) for child in tree['children']]
    if tree['kind']=='dict':return dict(zip(tree['keys'],items))
    return tuple(items) if tree['kind']=='tuple' else items


class BrowserPolicy:
    def __init__(self,manifest='policy_manifest.json',runner=None,record=True):
        self.manifest=json.loads(Path(manifest).read_text()) if not isinstance(manifest,dict) else manifest
        if self.manifest['checkpoint']!=CHECKPOINT or self.manifest['graph_count']!=25:raise ValueError('Frozen complete9a policy manifest required')
        self.runner=runner or self._browser_runner;self.record=record
        self.calls=0;self.seconds=0.;self.records=[];self.module_objects={};self.class_objects={}
        self.graphs={g['key']:g for g in self.manifest['graphs']}
        self.parameters={token:(_ParameterArray(spec['data'],spec['dtype'],token) if 'data' in spec else _OpaqueArray(token,spec)) for token,spec in self.manifest['arrays'].items()}
        _install_numpy_shim();self._load_modules();self.reset()

    @staticmethod
    def _browser_runner(graph_key,input_json):
        from js import gift_ort_run
        from pyodide.ffi import run_sync
        return run_sync(gift_ort_run(graph_key,input_json))

    def _run(self,key,args):
        graph=self.graphs[key];flat=list(_flatten(args))
        if len(flat)!=len(graph['inputs']):raise ValueError(('input tree differs',key,len(flat),len(graph['inputs'])))
        inputs=[]
        for spec,value in zip(graph['inputs'],flat):
            a=np.asarray(value,dtype=spec['type'])
            if list(a.shape)!=spec['dims']:raise ValueError(('input shape differs',key,spec['name'],a.shape,spec['dims']))
            inputs.append(dict(name=spec['name'],type=spec['type'],dims=spec['dims'],data=a.reshape(-1).tolist()))
        start=time.perf_counter();raw=self.runner(key,json.dumps(inputs,separators=(',',':')));self.seconds+=time.perf_counter()-start;self.calls+=1
        outputs=json.loads(raw) if isinstance(raw,str) else raw
        values=[]
        for spec,output in zip(graph['outputs'],outputs):
            if output['name']!=spec['name']:raise ValueError('ONNX output order differs')
            values.append(np.asarray(output['data'],dtype=spec['type']).reshape(spec['dims']))
        return _unflatten(graph['output_tree'],iter(values))

    def _prediction(self,spec):
        routes=spec['routes']
        def predict(parameters,*args):
            key=json.dumps(_signature(parameters),separators=(',',':'))
            if key not in routes:raise RuntimeError('No immutable ONNX graph for this exact parameter tree')
            return self._run(routes[key],args)
        return predict

    def _native(self,variables,carry,observation,reset):
        actor,bridge=self._resolve_function(self.manifest['bridge'])(observation)
        nc,logits,value,learned=self._run('native_actor.onnx',(carry,actor,reset))
        action,audit=self._resolve_function(self.manifest['selector'])(logits,observation,reset,learned['selected'])
        if not bool(audit['execution_valid']):raise ValueError('Invalid native action history/logits; refusing physics step')
        return nc,action,value,logits,learned,audit,bridge

    def _resolve_function(self,key):
        module,name=self.manifest['functions'][key]
        return self.module_objects[module].__dict__[name]

    def _decode(self,value,objects=None):
        if isinstance(value,list):return [self._decode(v,objects) for v in value]
        if not isinstance(value,dict):return value
        if '$tuple' in value:return tuple(self._decode(v,objects) for v in value['$tuple'])
        if '$array' in value:return self.parameters[value['$array']]
        if '$inline_array' in value:return np.asarray(value['$inline_array'],dtype=value['dtype'])
        if '$inline_arrays' in value:return tuple(np.asarray(v['data'],dtype=v['dtype']) for v in value['$inline_arrays'])
        if '$prediction' in value:return self._prediction(value['$prediction'])
        if '$native' in value:return self._native
        if '$function' in value:return self._resolve_function(value['$function'])
        if '$object' in value:return objects[value['$object']]
        return {k:self._decode(v,objects) for k,v in value.items()}

    def _load_modules(self):
        def make(name):
            if name in self.module_objects:return self.module_objects[name]
            if name in sys.modules and not name.startswith(('research','envs')):return sys.modules[name]
            m=types.ModuleType(name);m.__package__=name.rpartition('.')[0];m.__path__=[]
            m.__dict__['_numpy_set']=_numpy_set;m.__dict__['_numpy_binary']=_numpy_binary
            self.module_objects[name]=m;sys.modules[name]=m
            if '.' in name:parent,leaf=name.rsplit('.',1);setattr(make(parent),leaf,m)
            return m
        for name in self.manifest['modules']:make(name)
        for name,spec in self.manifest['modules'].items():
            m=make(name)
            for key,value in spec['values'].items():m.__dict__[key]=self._decode(value)
        # All function bodies are compiled first; calls resolve aliases afterward.
        for name,spec in self.manifest['modules'].items():
            m=make(name)
            for entry in spec['functions'].values():
                try:exec(_compile(entry['source'],f'<policy:{name}>'),m.__dict__)
                except NameError as error:
                    # Default argument constants/imports must be bound before def.
                    self._bind_aliases(name,soft=True)
                    exec(_compile(entry['source'],f'<policy:{name}>'),m.__dict__)
        for name in self.manifest['modules']:self._bind_aliases(name,soft=True)
        for spec in self.manifest['classes']:
            m=make(spec['module'])
            if 'namedtuple' in spec:cls=collections.namedtuple(spec['name'],spec['namedtuple'])
            else:
                bases=[]
                for i,base in enumerate(spec['bases']):
                    obj=self.class_objects[base['class']] if 'class' in base else self._import_attr(base['import'],base['name'])
                    symbol='_base_'+str(i);m.__dict__[symbol]=obj;bases.append(symbol)
                header='class _browser_'+spec['key']+'('+','.join(bases or ['object'])+'):'
                body='\n'.join('    '+line for method in spec['methods'] for line in method.splitlines()) or '    pass'
                exec(_compile(header+'\n'+body,f'<class:{spec["module"]}:{spec["name"]}>'),m.__dict__)
                cls=m.__dict__['_browser_'+spec['key']]
            self.class_objects[spec['key']]=cls
        for name in self.manifest['modules']:self._bind_aliases(name,soft=False)

    @staticmethod
    def _import_attr(module,name):
        obj=importlib.import_module(module)
        for part in name.split('.'):obj=getattr(obj,part)
        return obj

    def _bind_aliases(self,name,soft=False):
        m=self.module_objects[name]
        for key,spec in self.manifest['modules'][name]['aliases'].items():
            try:
                if 'module' in spec:value=self.module_objects.get(spec['module']) or importlib.import_module(spec['module'])
                elif 'function' in spec:value=self._resolve_function(spec['function'])
                elif 'class' in spec:value=self.class_objects[spec['class']]
                elif 'module_attr' in spec:value=getattr(self.module_objects[spec['module_attr'][0]],spec['module_attr'][1])
                else:value=self._import_attr(spec['import'],spec['name'])
                m.__dict__[key]=value
            except (KeyError,AttributeError):
                if not soft:raise

    def reset(self):
        objects={}
        for key,spec in self.manifest['objects'].items():
            cls=_Policy if spec['cls']=='policy' else self.class_objects[spec['cls']]
            objects[key]=cls.__new__(cls)
        for key,spec in self.manifest['objects'].items():
            for name,value in spec['fields'].items():object.__setattr__(objects[key],name,self._decode(value,objects))
        self.stepper=self._decode(self.manifest['root_object'],objects);self.objects=objects
        self.carry=np.asarray(self.manifest['initial_carry'],np.float32).copy();self.control=0;self.records=[]
        return self.carry.copy()

    def step(self,observation,carry=None,reset=None,context=None):
        observation=np.asarray(observation,np.float32)
        carry=self.carry if carry is None else np.asarray(carry,np.float32)
        reset=np.asarray(self.control==0 if reset is None else reset,dtype=bool)
        context=np.zeros(36,np.float32) if context is None else np.asarray(context,np.float32)
        if observation.shape!=(21,) or carry.shape!=(1,130) or context.shape!=(36,):raise ValueError('Policy ABI is observation21/carry130/context36')
        self.stepper.context_provider=lambda obs:context
        value=self.stepper(None,carry,observation,reset)
        self.carry=np.asarray(value[0],np.float32);self.control+=1
        audit=value[5]
        if not bool(audit['execution_valid']):raise ValueError('Illegal action state; refusing dynamics')
        row=self.stepper.records[-1]
        if self.record:self.records.append(row)
        # Original execution reads only each layer's latest record. The separate
        # causal history and all three option counters remain intact.
        for obj in self.objects.values():
            records=vars(obj).get('records')
            if isinstance(records,list) and len(records)>1:del records[:-1]
        return value

    def report(self):
        return dict(backend='onnxruntime-web',checkpoint=CHECKPOINT,graph_count=25,onnx_calls=self.calls,onnx_seconds=self.seconds,
            controls=self.control,whole_policy=True,jax_imported=False,server_inference=False,bitwise_equivalence_claim=False)
