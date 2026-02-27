'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'prerequisites', title: 'Prerequisites', level: 2 },
  { id: 'deployment', title: 'Deployment', level: 2 },
  { id: 'service', title: 'Service & Ingress', level: 2 },
  { id: 'gpu-scheduling', title: 'GPU Scheduling', level: 2 },
  { id: 'autoscaling', title: 'Autoscaling', level: 2 },
]

export default function KubernetesPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Kubernetes Deployment"
          description="Deploy ZSE on Kubernetes with GPU support, autoscaling, and high availability."
          badge="Deployment"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Kubernetes is the recommended platform for production ZSE deployments at scale. 
            This guide covers GPU scheduling, autoscaling, and high-availability configurations.
          </p>

          <CardGrid columns={3}>
            <Card
              title="GPU Nodes"
              description="NVIDIA device plugin integration"
            />
            <Card
              title="Autoscaling"
              description="HPA based on GPU metrics"
            />
            <Card
              title="High Availability"
              description="Multi-replica deployments"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="prerequisites" title="Prerequisites">
          <FeatureList features={[
            "Kubernetes cluster 1.25+",
            "NVIDIA GPU Operator or device plugin installed",
            "kubectl configured for your cluster",
            "Helm 3.x (optional, for chart installation)",
          ]} />

          <DocSubSection id="nvidia-plugin" title="NVIDIA Device Plugin">
            <CodeBlock
              language="bash"
              code={`# Install NVIDIA device plugin (if not using GPU Operator)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs are detected
kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\\t"}{.status.capacity.nvidia\\.com/gpu}{"\\n"}{end}'`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="deployment" title="Deployment">
          <DocSubSection id="basic-deployment" title="Basic Deployment">
            <CodeBlock
              language="yaml"
              filename="zse-deployment.yaml"
              code={`apiVersion: apps/v1
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
          claimName: zse-models-pvc`}
            />
          </DocSubSection>

          <DocSubSection id="pvc" title="Persistent Volume Claim">
            <CodeBlock
              language="yaml"
              filename="zse-pvc.yaml"
              code={`apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zse-models-pvc
spec:
  accessModes:
    - ReadOnlyMany  # Models are read-only
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi`}
            />
          </DocSubSection>

          <DocSubSection id="configmap" title="ConfigMap">
            <CodeBlock
              language="yaml"
              filename="zse-configmap.yaml"
              code={`apiVersion: v1
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
      format: json`}
            />
          </DocSubSection>

          <DocSubSection id="apply" title="Apply Resources">
            <CodeBlock
              language="bash"
              code={`kubectl apply -f zse-pvc.yaml
kubectl apply -f zse-configmap.yaml
kubectl apply -f zse-deployment.yaml

# Check status
kubectl get pods -l app=zse
kubectl logs -f deployment/zse`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="service" title="Service & Ingress">
          <DocSubSection id="service" title="Service">
            <CodeBlock
              language="yaml"
              filename="zse-service.yaml"
              code={`apiVersion: v1
kind: Service
metadata:
  name: zse
spec:
  selector:
    app: zse
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP`}
            />
          </DocSubSection>

          <DocSubSection id="ingress" title="Ingress with TLS">
            <CodeBlock
              language="yaml"
              filename="zse-ingress.yaml"
              code={`apiVersion: networking.k8s.io/v1
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
              number: 80`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="gpu-scheduling" title="GPU Scheduling">
          <DocSubSection id="node-selector" title="Node Selector">
            <CodeBlock
              language="yaml"
              code={`spec:
  template:
    spec:
      nodeSelector:
        gpu-type: a100  # Select specific GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule`}
            />
          </DocSubSection>

          <DocSubSection id="affinity" title="Pod Affinity">
            <CodeBlock
              language="yaml"
              code={`spec:
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
              topologyKey: kubernetes.io/hostname`}
            />

            <Callout type="info">
              Pod anti-affinity spreads replicas across nodes for better fault tolerance 
              and GPU utilization.
            </Callout>
          </DocSubSection>

          <DocSubSection id="multi-gpu" title="Multi-GPU Pods">
            <CodeBlock
              language="yaml"
              code={`resources:
  limits:
    nvidia.com/gpu: 2  # Request 2 GPUs per pod
  requests:
    memory: "64Gi"
    cpu: "16"`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="autoscaling" title="Autoscaling">
          <DocSubSection id="hpa" title="Horizontal Pod Autoscaler">
            <CodeBlock
              language="yaml"
              filename="zse-hpa.yaml"
              code={`apiVersion: autoscaling/v2
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
        averageValue: "100"`}
            />
          </DocSubSection>

          <DocSubSection id="gpu-metrics" title="GPU-Based Scaling">
            <p className="mb-4">
              For GPU-based autoscaling, use the DCGM exporter with Prometheus adapter:
            </p>

            <CodeBlock
              language="yaml"
              code={`metrics:
- type: External
  external:
    metric:
      name: DCGM_FI_DEV_GPU_UTIL
      selector:
        matchLabels:
          pod: zse
    target:
      type: AverageValue
      averageValue: "80"  # Scale when GPU > 80%`}
            />
          </DocSubSection>

          <Callout type="warning">
            GPU autoscaling requires available GPU capacity in your cluster. 
            Configure cluster autoscaler to add GPU nodes when needed.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/deployment/docker', title: 'Docker' }}
          next={{ href: '/docs/deployment/serverless', title: 'Serverless' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
