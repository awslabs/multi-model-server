/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.archive;

public class InvalidModelException extends Exception {

    static final long serialVersionUID = 1L;
    private final String errorCode;

    public InvalidModelException(String code, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = code;
    }

    public InvalidModelException(String code, String message) {
        super(message);
        this.errorCode = code;
    }

    /**
     * Constructs a new {@code InvalidModelException} with the specified detail message and cause.
     *
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method).
     */
    public InvalidModelException(String message) {
        super(message);
        this.errorCode = "";
    }

    public String getErrorCode() {
        return errorCode;
    }
}
