package com.amazonaws.ml.mms.common;

public interface ErrorCodes {
    String UNKNOWN_ERROR = "UnknownError";
    String INVALID_URI = "InvalidURI";
    String MESSAGE_DECODE_FAILURE = "MessageDecodeFailure";
    String WORKER_INSTANTIATION_ERROR = "WorkerInstantiationError";
    String MODELS_POST_INVALID_REQUEST = "RegisterApi.InvalidRequest";
    String MODELS_POST_MODEL_MANIFEST_RUNTIME_INVALID = "RegisterApi.ModelManifestRuntimeInvalid";
    String MODELS_POST_MODEL_ALREADY_REGISTERED = "RegisterApi.ModelAlreadyRegistered";

    String PREDICTIONS_API_INVALID_REQUEST = "PredictionsApi.InvalidRequest";
    String PREDICTIONS_API_MODEL_NOT_REGISTERED = "PredictionsApi.ModelNotRegistered";
    String PREDICTIONS_API_INVALID_PARAMETERS = "PredictionsApi.InvalidParameters";
    String PREDICTIONS_API_MODEL_NOT_SCALED = "PredictionsApi.ModelNotScaled";

    String LIST_MODELS_INVALID_REQUEST_HEADER = "ListModelApi.InvalidHttpMethod";

    String MODELS_API_INVALID_MODELS_REQUEST = "ModelsApi.InvalidRequest";
    String MODELS_API_MODEL_NOT_FOUND = "ModelsApi.ModelNotFound";

    String INTERNAL_SERVER_ERROR_BACKEND_WORKER_INSTANTIATION =
            "InternalServerError.BackendWorkerInstantiationError";
    String INTERNAL_SERVER_ERROR_WORKER_HEALTH_CHECK_TIMEOUT =
            "InternalServerError.WorkerHealthCheckTimeout";
    String INTERNAL_SERVER_ERROR_WORKER_LISTEN_FAILURE = "InternalServerError.WorkerListenFailure";
}
