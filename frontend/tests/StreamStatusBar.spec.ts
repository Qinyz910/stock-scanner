import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import StreamStatusBar from '@/components/StreamStatusBar.vue';

describe('StreamStatusBar', () => {
  it('emits cancel event when cancel button is clicked', async () => {
    const wrapper = mount(StreamStatusBar, {
      props: {
        isStreaming: true,
        processed: 2,
        total: 5,
        chunkCount: 3,
        retries: 1,
        statusText: '接收中...',
        canCancel: true,
      },
      global: {
        stubs: {
          'n-progress': {
            template: '<div class="n-progress"><slot /></div>',
            props: ['type', 'percentage', 'status', 'processing'],
          },
          'n-button': {
            template: '<button @click="$emit(\'click\')"><slot /></button>',
            props: ['size', 'tertiary', 'type'],
          },
          'n-space': {
            template: '<div class="n-space"><slot /></div>',
            props: ['size'],
          },
          'n-text': {
            template: '<span class="n-text"><slot /></span>',
            props: ['size'],
          },
        },
      },
    });

    expect(wrapper.text()).toContain('接收中');
    await wrapper.find('button').trigger('click');
    expect(wrapper.emitted('cancel')).toBeTruthy();
  });
});
