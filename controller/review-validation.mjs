/** Copyable, bounded-schema diagnostics for native review tool submissions. */
export class ReviewValidationError extends Error {
 constructor(message,details){
  super(message+'\nValidation details: '+JSON.stringify(details));
  this.name='ReviewValidationError';this.code='INVALID_REVIEW_SUBMISSION';this.details=details;
 }
}
export function validateReviewReferences(refs,allowed,{field,message,required=false,...context}){
 const known=[...new Set(allowed)],invalidRefs=Array.isArray(refs)?[...new Set(refs.filter(ref=>!known.includes(ref)))]:[];
 if(!Array.isArray(refs)||required&&!refs.length||invalidRefs.length)
  throw new ReviewValidationError(message,{field,...context,invalidRefs,allowedRefs:known,required,arrayRequired:true});
}
