package com.amazonaws.ml.mms.archive;

public class InvalidModelException extends Exception {

    static final long serialVersionUID = 1L;

    /** Creates a new {@code InvalidModelException} instance. */
    public InvalidModelException() {
        super();
    }

    /**
     * Constructs a new {@code InvalidModelException} with the specified detail message and cause.
     *
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method).
     */
    public InvalidModelException(String message) {
        super(message);
    }

    /**
     * Constructs a new {@code InvalidModelException} with the specified detail message and cause.
     *
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method).
     * @param cause the cause (which is saved for later retrieval by the {@link #getCause()}
     *     method). (A <tt>null</tt> value is permitted, and indicates that the cause is nonexistent
     *     or unknown.)
     */
    public InvalidModelException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new {@code InvalidModelException} with the specified detail message and cause.
     *
     * @param cause the cause (which is saved for later retrieval by the {@link #getCause()}
     *     method). (A <tt>null</tt> value is permitted, and indicates that the cause is nonexistent
     *     or unknown.)
     */
    public InvalidModelException(Throwable cause) {
        super(cause);
    }
}
