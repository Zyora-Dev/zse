(()=>{var e={};e.id=350,e.ids=[350],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},6503:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>r.a,__next_app__:()=>m,originalPathname:()=>p,pages:()=>d,routeModule:()=>u,tree:()=>c});var i=s(482),a=s(9108),l=s(2563),r=s.n(l),n=s(8300),o={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(o[e]=()=>n[e]);s.d(t,o);let c=["",{children:["docs",{children:["deployment",{children:["kubernetes",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,9771)),"/Users/redfoxhotels/zse/website/src/app/docs/deployment/kubernetes/page.tsx"]}]},{}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/deployment/kubernetes/page.tsx"],p="/docs/deployment/kubernetes/page",m={require:s,loadChunk:()=>Promise.resolve()},u=new i.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/docs/deployment/kubernetes/page",pathname:"/docs/deployment/kubernetes",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:c}})},7286:(e,t,s)=>{Promise.resolve().then(s.bind(s,8268))},8268:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>c});var i=s(5344),a=s(1499),l=s(196),r=s(1812),n=s(9039);let o=[{id:"overview",title:"Overview",level:2},{id:"prerequisites",title:"Prerequisites",level:2},{id:"deployment",title:"Deployment",level:2},{id:"service",title:"Service & Ingress",level:2},{id:"gpu-scheduling",title:"GPU Scheduling",level:2},{id:"autoscaling",title:"Autoscaling",level:2}];function c(){return(0,i.jsxs)("div",{className:"flex",children:[(0,i.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[i.jsx(a.lv,{title:"Kubernetes Deployment",description:"Deploy ZSE on Kubernetes with GPU support, autoscaling, and high availability.",badge:"Deployment"}),(0,i.jsxs)(a.Je,{id:"overview",title:"Overview",children:[i.jsx("p",{className:"mb-4",children:"Kubernetes is the recommended platform for production ZSE deployments at scale. This guide covers GPU scheduling, autoscaling, and high-availability configurations."}),(0,i.jsxs)(n.gy,{columns:3,children:[i.jsx(n.Zb,{title:"GPU Nodes",description:"NVIDIA device plugin integration"}),i.jsx(n.Zb,{title:"Autoscaling",description:"HPA based on GPU metrics"}),i.jsx(n.Zb,{title:"High Availability",description:"Multi-replica deployments"})]})]}),(0,i.jsxs)(a.Je,{id:"prerequisites",title:"Prerequisites",children:[i.jsx(n.VS,{features:["Kubernetes cluster 1.25+","NVIDIA GPU Operator or device plugin installed","kubectl configured for your cluster","Helm 3.x (optional, for chart installation)"]}),i.jsx(a.KU,{id:"nvidia-plugin",title:"NVIDIA Device Plugin",children:i.jsx(l.d,{language:"bash",code:`# Install NVIDIA device plugin (if not using GPU Operator)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs are detected
kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\\t"}{.status.capacity.nvidia\\.com/gpu}{"\\n"}{end}'`})})]}),(0,i.jsxs)(a.Je,{id:"deployment",title:"Deployment",children:[i.jsx(a.KU,{id:"basic-deployment",title:"Basic Deployment",children:i.jsx(l.d,{language:"yaml",filename:"zse-deployment.yaml",code:`apiVersion: apps/v1
kind: Deployment
metadata:
  name: zse
  labels:
    app: zse
spec:
  replicas: 2
  selector:
    matchLabels:
      app: zse
  template:
    metadata:
      labels:
        app: zse
    spec:
      containers:
      - name: zse
        image: zyora/zse:latest
        ports:
        - containerPort: 8000
        command: ["zse", "serve", "/models/qwen7b.zse"]
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /models
        env:
        - name: ZSE_HOST
          value: "0.0.0.0"
        - name: ZSE_PORT
          value: "8000"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: zse-models-pvc`})}),i.jsx(a.KU,{id:"pvc",title:"Persistent Volume Claim",children:i.jsx(l.d,{language:"yaml",filename:"zse-pvc.yaml",code:`apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zse-models-pvc
spec:
  accessModes:
    - ReadOnlyMany  # Models are read-only
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi`})}),i.jsx(a.KU,{id:"configmap",title:"ConfigMap",children:i.jsx(l.d,{language:"yaml",filename:"zse-configmap.yaml",code:`apiVersion: v1
kind: ConfigMap
metadata:
  name: zse-config
data:
  config.yaml: |
    server:
      host: 0.0.0.0
      port: 8000
      workers: 4
    model:
      max_context: 8192
      batch_size: 32
    logging:
      level: info
      format: json`})}),i.jsx(a.KU,{id:"apply",title:"Apply Resources",children:i.jsx(l.d,{language:"bash",code:`kubectl apply -f zse-pvc.yaml
kubectl apply -f zse-configmap.yaml
kubectl apply -f zse-deployment.yaml

# Check status
kubectl get pods -l app=zse
kubectl logs -f deployment/zse`})})]}),(0,i.jsxs)(a.Je,{id:"service",title:"Service & Ingress",children:[i.jsx(a.KU,{id:"service",title:"Service",children:i.jsx(l.d,{language:"yaml",filename:"zse-service.yaml",code:`apiVersion: v1
kind: Service
metadata:
  name: zse
spec:
  selector:
    app: zse
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP`})}),i.jsx(a.KU,{id:"ingress",title:"Ingress with TLS",children:i.jsx(l.d,{language:"yaml",filename:"zse-ingress.yaml",code:`apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: zse
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: zse-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: zse
            port:
              number: 80`})})]}),(0,i.jsxs)(a.Je,{id:"gpu-scheduling",title:"GPU Scheduling",children:[i.jsx(a.KU,{id:"node-selector",title:"Node Selector",children:i.jsx(l.d,{language:"yaml",code:`spec:
  template:
    spec:
      nodeSelector:
        gpu-type: a100  # Select specific GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule`})}),(0,i.jsxs)(a.KU,{id:"affinity",title:"Pod Affinity",children:[i.jsx(l.d,{language:"yaml",code:`spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: zse
              topologyKey: kubernetes.io/hostname`}),i.jsx(r.U,{type:"info",children:"Pod anti-affinity spreads replicas across nodes for better fault tolerance and GPU utilization."})]}),i.jsx(a.KU,{id:"multi-gpu",title:"Multi-GPU Pods",children:i.jsx(l.d,{language:"yaml",code:`resources:
  limits:
    nvidia.com/gpu: 2  # Request 2 GPUs per pod
  requests:
    memory: "64Gi"
    cpu: "16"`})})]}),(0,i.jsxs)(a.Je,{id:"autoscaling",title:"Autoscaling",children:[i.jsx(a.KU,{id:"hpa",title:"Horizontal Pod Autoscaler",children:i.jsx(l.d,{language:"yaml",filename:"zse-hpa.yaml",code:`apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zse
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zse
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"`})}),(0,i.jsxs)(a.KU,{id:"gpu-metrics",title:"GPU-Based Scaling",children:[i.jsx("p",{className:"mb-4",children:"For GPU-based autoscaling, use the DCGM exporter with Prometheus adapter:"}),i.jsx(l.d,{language:"yaml",code:`metrics:
- type: External
  external:
    metric:
      name: DCGM_FI_DEV_GPU_UTIL
      selector:
        matchLabels:
          pod: zse
    target:
      type: AverageValue
      averageValue: "80"  # Scale when GPU > 80%`})]}),i.jsx(r.U,{type:"warning",children:"GPU autoscaling requires available GPU capacity in your cluster. Configure cluster autoscaler to add GPU nodes when needed."})]}),i.jsx(a.KO,{prev:{href:"/docs/deployment/docker",title:"Docker"},next:{href:"/docs/deployment/serverless",title:"Serverless"}})]}),i.jsx(a.o5,{items:o})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>n,VS:()=>o,Zb:()=>c,gy:()=>d});var i=s(5344),a=s(1912),l=s(2312),r=s(1453);function n({steps:e}){return i.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,i.jsxs)(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&i.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),i.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:i.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,i.jsxs)("div",{children:[i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&i.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&i.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:i.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&i.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function o({features:e}){return i.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,i.jsxs)(a.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[i.jsx(l.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),i.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function c({title:e,description:t,icon:s,href:l,children:n}){let o=l?"a":"div";return i.jsx(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:l?{y:-2}:void 0,children:(0,i.jsxs)(o,{...l?{href:l,className:"block"}:{},className:(0,r.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",l&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&i.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:i.jsx(s,{className:"w-4 h-4 text-lime"})}),i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&i.jsx("p",{className:"text-sm text-white/50",children:t}),n]})})}function d({children:e,columns:t=2}){return i.jsx("div",{className:(0,r.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},9771:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>l,__esModule:()=>a,default:()=>r});let i=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/deployment/kubernetes/page.tsx`),{__esModule:a,$$typeof:l}=i,r=i.default}};var t=require("../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),i=t.X(0,[638,498,697,224,782,883],()=>s(6503));module.exports=i})();