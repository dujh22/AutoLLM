import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { App, Button, Modal, Form, Input } from 'antd';
import styles from './index.module.scss';
import { localPopupHandler } from '@/utils/common.ts';
import request from '@/utils/request.ts';
import { REQUEST_URLS } from '@/config/requet-urls.ts';
import { ROUTE_URLS } from '@/config/route-urls.ts';
import router from '@/config/routes';
import { useAiStore } from '@/store';

export interface IFieldType {
  phone: string;
  intro: string;
  paper: string;
  publicAccount: string;
  news: string;
}

export interface IReqParams {
  phone_number: string;
  prompt: string;
  paper_link: string[];
  zzh_link: string[];
  news_link: string[];
}

interface IProps {
  type: 'new' | 'edit';
  visible: boolean;
  setVisible: (vis: boolean) => void;
}

const SetupModal: React.FC<IProps> = ({ visible, setVisible, type }) => {
  const { message } = App.useApp();

  const phone = useAiStore(s => s.phone);
  const setPhone = useAiStore(s => s.setPhone);

  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);

  const [form] = Form.useForm<IFieldType>();

  const onFinish = useCallback(() => {
    if (loading) {
      return;
    }
    setLoading(true);
    form.validateFields().then(() => {
      const { phone, intro, paper, publicAccount, news } = form.getFieldsValue();
      const params: IReqParams = {
        phone_number: phone,
        prompt: intro,
        paper_link: paper.split('\n').filter(Boolean),
        zzh_link: publicAccount.split('\n').filter(Boolean),
        news_link: news ? news.split('\n').filter(Boolean) : [],
      };
      request
        .post(REQUEST_URLS.saveProfile, params)
        .then(res => {
          if (res.status === 200) {
            setPhone(phone);
            localPopupHandler.set(true);
            router.navigate(ROUTE_URLS.home);
            setVisible(false);
            message.success({
              content: '保存成功',
              duration: 1,
            });
            return;
          }
          return Promise.reject(res.statusText);
        })
        .catch(e => {
          message.error(typeof e === 'string' ? e || '网络错误' : e.toString());
        })
        .finally(() => {
          setLoading(false);
        });
    });
  }, [form, loading, message, setPhone, setVisible]);

  const Step0 = useMemo(
    () => (
      <div className={styles.contentContainer}>
        <div className={styles.introList}>
          <div className={styles.introItem}>
            <div className={styles.text}>知识渠道获取</div>
            <div className={styles.image} />
          </div>
          <div className={styles.introItem}>
            <div className={styles.text}>知识自动整理</div>
            <div className={styles.image} />
          </div>
          <div className={styles.introItem}>
            <div className={styles.text}>新知识提取与分享</div>
            <div className={styles.image} />
          </div>
        </div>
        <div className={styles.foot}>
          <Button type="primary" onClick={() => setStep(1)}>
            开始
          </Button>
        </div>
      </div>
    ),
    [setStep]
  );

  const Step1 = useMemo(() => {
    return (
      <div className={styles.contentContainer}>
        <Form name="form" form={form} layout="vertical" className={styles.form}>
          <Form.Item<IFieldType> name="phone" label="手机号" required>
            <Input size="large" allowClear placeholder="请输入手机号" disabled={type === 'edit'} />
          </Form.Item>
          <Form.Item<IFieldType> name="intro" label="个人知识偏好" required>
            <Input.TextArea
              size="large"
              autoSize={{ maxRows: 3 }}
              allowClear
              placeholder="例如：我关心知识图谱和大模型的知识，希望在学习过程中，可以重点关注此类知识"
            />
          </Form.Item>
          <Form.Item<IFieldType> name="paper" label="论文" required>
            <Input.TextArea size="large" autoSize={{ maxRows: 3 }} allowClear placeholder="请添加关注的论文链接，按回车输入多个链接" />
          </Form.Item>
          <Form.Item<IFieldType> name="publicAccount" label="公众号" required>
            <Input.TextArea
              size="large"
              autoSize={{ maxRows: 3 }}
              allowClear
              placeholder="请添加关注的公众号文章链接，按回车输入多个链接"
            />
          </Form.Item>
          <Form.Item<IFieldType> name="news" label="新闻">
            <Input.TextArea size="large" autoSize={{ maxRows: 3 }} allowClear placeholder="请添加关注的新闻链接，按回车输入多个链接" />
          </Form.Item>
        </Form>
        <div className={styles.foot}>
          <Button type="primary" onClick={onFinish} loading={loading}>
            完成
          </Button>
        </div>
      </div>
    );
  }, [form, loading, onFinish, type]);

  const ContentEle = useMemo(() => {
    if (step === 0) {
      return Step0;
    }
    return Step1;
  }, [Step0, Step1, step]);

  useEffect(() => {
    const popupShown = localPopupHandler.get();
    if (!popupShown) {
      setVisible(true);
    }
  }, [setVisible]);

  useEffect(() => {
    if (visible) {
      setStep(type === 'edit' ? 1 : 0);
      if (type === 'edit') {
        request
          .get(REQUEST_URLS.getProfile, {
            params: {
              phone_number: phone,
            },
          })
          .then(res => {
            const params: IReqParams = res.data;
            const formData: IFieldType = {
              phone: params.phone_number,
              intro: params.prompt,
              publicAccount: params.zzh_link.join('\n'),
              paper: params.paper_link.join('\n'),
              news: params.news_link.join('\n'),
            };
            form.setFieldsValue(formData);
          });
      }
    } else {
      form.resetFields();
    }
  }, [form, phone, type, visible]);

  return (
    <Modal
      width={800}
      open={visible}
      title="请设置基本信息"
      onClose={() => setVisible(false)}
      closeIcon={type === 'new' ? null : undefined}
      footer={null}
      onCancel={() => setVisible(false)}
    >
      {ContentEle}
    </Modal>
  );
};

export default SetupModal;
