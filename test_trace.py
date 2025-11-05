import os
import time
from opentelemetry import trace, propagate
from opentelemetry.sdk.resources import Resource # <-- Import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator

# 1. Configure OpenTelemetry SDK
SERVICE_NAME = "EC2-Basic-Test-Service"

# CORRECTED: Create an explicit Resource object
resource_attributes = {"service.name": SERVICE_NAME}
resource = Resource.create(resource_attributes) # <-- Use Resource.create()

# Configure the tracer to use the AWS X-Ray ID generator for proper formatting
provider = TracerProvider(
    id_generator=AwsXRayIdGenerator(),
    resource=resource # <-- Pass the created Resource object
)

# Configure the OTLP exporter to send traces to the ADOT Collector (http://localhost:4317)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)

# Set up the span processor to send traces in batches
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)
trace.set_tracer_provider(provider)

# Set the global propagator for context transfer (highly recommended for X-Ray)
from opentelemetry.propagators.aws.aws_xray_propagator import AwsXRayPropagator
propagate.set_global_textmap(AwsXRayPropagator())

tracer = trace.get_tracer(__name__)

# 2. Generate a Trace
print(f"Generating a trace for service: {SERVICE_NAME}")
with tracer.start_as_current_span("MainOperation") as span:
    span.set_attribute("http.method", "GET")
    span.set_attribute("test.env", "EC2")

    # Create a sub-span/subsegment
    with tracer.start_as_current_span("SubFunction"):
        time.sleep(0.1)
        print("  -> Sub-span complete.")

    time.sleep(0.2)
    span.set_attribute("result.status", "Success")

# Force flush the exporter to ensure data is sent immediately
provider.force_flush()
print("Trace sent to ADOT Collector at localhost:4317.")