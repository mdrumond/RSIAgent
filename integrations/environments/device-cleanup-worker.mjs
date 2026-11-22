import {drainDeviceCleanup} from './device-cleanup.mjs';
await drainDeviceCleanup(process.argv[2],process.argv[3]);
