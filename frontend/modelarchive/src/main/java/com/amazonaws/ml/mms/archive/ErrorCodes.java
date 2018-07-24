package com.amazonaws.ml.mms.archive;

public interface ErrorCodes {
    String UNKNOWN_ERROR = "UnknownError";
    String MODEL_ARCHIVE_DOWNLOAD_FAIL = "ModelArchiveDownloadFail";
    String INVALID_URL = "InvalidUrl";
    String MODEL_NOT_FOUND = "ModelNotFound";
    String MODEL_ARCHIVE_INCORRECT = "ModelArchiveIncorrect";
    String MISSING_ARTIFACT_MANIFEST = "MissingArtifact.Manifest";
    String MISSING_ARTIFACT_SIGNATURE = "MissingArtifact.Signature";
    String INCORRECT_ARTIFACT_MANIFEST = "IncorrectArtifact.Manifest";
    String INCORRECT_ARTIFACT_SIGNATURE = "IncorrectArtifact.Signature";
    String INCORRECT_ARTIFACT_LEGACY_MODEL = "IncorrectArtifact.LegacyModel";
}
