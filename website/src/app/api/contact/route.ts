import { NextRequest, NextResponse } from 'next/server'

const ZEPTO_API_URL = 'https://api.zeptomail.in/v1.1/email'
const ZEPTO_TOKEN = 'Zoho-enczapikey PHtE6r0PQO/j2WEr+kBT7KC7H5P3Z98o+uxgfwAW444XWKIGSU1Vr9ktlGC/oxwsU/BCHP+byIw7sO+c4L/TLWu4YGYfWmqyqK3sx/VYSPOZsbq6x00btlQScULdUo7pc99o0ifVud/cNA=='
const MAIL_AGENT_ALIAS = '5e5a9f42cd45806f'
const ADMIN_EMAIL = 'info@zyoralabs.com'
const SENDER_EMAIL = 'info@zyoralabs.com'
const SENDER_NAME = 'ZSE by Zlabs'

interface ContactFormData {
  name: string
  email: string
  subject: string
  message: string
}

const subjectLabels: Record<string, string> = {
  general: 'General Inquiry',
  support: 'Technical Support',
  enterprise: 'Enterprise / Partnership',
  feedback: 'Feedback / Suggestions',
  other: 'Other',
}

function getAdminEmailTemplate(data: ContactFormData): string {
  const subjectLabel = subjectLabels[data.subject] || data.subject
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #000000;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #0a0a0a; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
          <!-- Header -->
          <tr>
            <td style="padding: 40px 40px 20px 40px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">
              <h1 style="margin: 0; color: #c0ff71; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                New Contact Form Submission
              </h1>
              <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.5); font-size: 14px;">
                ZSE Website Contact Form
              </p>
            </td>
          </tr>
          
          <!-- Content -->
          <tr>
            <td style="padding: 30px 40px;">
              <!-- Sender Info -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 24px;">
                <tr>
                  <td style="padding: 20px; background-color: rgba(192,255,113,0.05); border-radius: 12px; border: 1px solid rgba(192,255,113,0.1);">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                      <tr>
                        <td width="50%" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">From</p>
                          <p style="margin: 4px 0 0 0; color: #ffffff; font-size: 16px; font-weight: 600;">${data.name}</p>
                        </td>
                        <td width="50%" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Email</p>
                          <p style="margin: 4px 0 0 0; color: #c0ff71; font-size: 16px;">
                            <a href="mailto:${data.email}" style="color: #c0ff71; text-decoration: none;">${data.email}</a>
                          </p>
                        </td>
                      </tr>
                      <tr>
                        <td colspan="2" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Subject</p>
                          <p style="margin: 4px 0 0 0; color: #ffffff; font-size: 16px;">${subjectLabel}</p>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
              
              <!-- Message -->
              <div style="margin-bottom: 24px;">
                <p style="margin: 0 0 12px 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Message</p>
                <div style="padding: 20px; background-color: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                  <p style="margin: 0; color: rgba(255,255,255,0.85); font-size: 15px; line-height: 1.7; white-space: pre-wrap;">${data.message}</p>
                </div>
              </div>
              
              <!-- Reply Button -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td align="center">
                    <a href="mailto:${data.email}?subject=Re: ${subjectLabel}" style="display: inline-block; padding: 14px 32px; background-color: #c0ff71; color: #000000; font-size: 14px; font-weight: 600; text-decoration: none; border-radius: 8px;">
                      Reply to ${data.name}
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
              <p style="margin: 0; color: rgba(255,255,255,0.4); font-size: 13px;">
                This email was sent from the ZSE website contact form
              </p>
              <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.3); font-size: 12px;">
                © ${new Date().getFullYear()} Zyora Labs. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`
}

function getUserAutoReplyTemplate(data: ContactFormData): string {
  const subjectLabel = subjectLabels[data.subject] || data.subject
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #000000;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #0a0a0a; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
          <!-- Header with Logo -->
          <tr>
            <td style="padding: 40px 40px 30px 40px; text-align: center;">
              <div style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, rgba(192,255,113,0.15) 0%, rgba(192,255,113,0.05) 100%); border-radius: 12px; border: 1px solid rgba(192,255,113,0.2);">
                <span style="color: #c0ff71; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">zLLM</span>
                <span style="color: rgba(255,255,255,0.6); font-size: 24px; font-weight: 300;"> | </span>
                <span style="color: #ffffff; font-size: 24px; font-weight: 700;">ZSE</span>
              </div>
            </td>
          </tr>
          
          <!-- Main Content -->
          <tr>
            <td style="padding: 0 40px 40px 40px;">
              <!-- Greeting -->
              <h1 style="margin: 0 0 20px 0; color: #ffffff; font-size: 28px; font-weight: 700; text-align: center;">
                Thank you for reaching out!
              </h1>
              
              <p style="margin: 0 0 24px 0; color: rgba(255,255,255,0.7); font-size: 16px; line-height: 1.7; text-align: center;">
                Hi <strong style="color: #ffffff;">${data.name}</strong>, we've received your message and our team will get back to you as soon as possible.
              </p>
              
              <!-- Summary Card -->
              <div style="padding: 24px; background-color: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 32px;">
                <p style="margin: 0 0 16px 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Your Message Summary</p>
                
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                      <span style="color: rgba(255,255,255,0.5); font-size: 14px;">Subject:</span>
                      <span style="color: #ffffff; font-size: 14px; margin-left: 8px;">${subjectLabel}</span>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding: 12px 0 0 0;">
                      <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.5); font-size: 14px;">Message:</p>
                      <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 14px; line-height: 1.6; white-space: pre-wrap;">${data.message.substring(0, 300)}${data.message.length > 300 ? '...' : ''}</p>
                    </td>
                  </tr>
                </table>
              </div>
              
              <!-- Response Time -->
              <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(192,255,113,0.08) 0%, rgba(192,255,113,0.02) 100%); border-radius: 12px; border: 1px solid rgba(192,255,113,0.15); margin-bottom: 32px;">
                <p style="margin: 0; color: #c0ff71; font-size: 14px; font-weight: 600;">
                  ⚡ Expected Response Time: Within 24-48 hours
                </p>
              </div>
              
              <!-- Quick Links -->
              <p style="margin: 0 0 16px 0; color: rgba(255,255,255,0.5); font-size: 14px; text-align: center;">
                While you wait, explore our resources:
              </p>
              
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0">
                      <tr>
                        <td style="padding: 0 8px;">
                          <a href="https://zse.dev/docs" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            📚 Documentation
                          </a>
                        </td>
                        <td style="padding: 0 8px;">
                          <a href="https://github.com/zse/zse" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            ⭐ GitHub
                          </a>
                        </td>
                        <td style="padding: 0 8px;">
                          <a href="https://discord.gg/f9JKreJA" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            💬 Discord
                          </a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
              <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.6); font-size: 14px;">
                <strong style="color: #c0ff71;">ZSE</strong> by Zyora Labs
              </p>
              <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.4); font-size: 13px;">
                Fast LLM Inference Engine | 3.9s Cold Starts
              </p>
              <p style="margin: 12px 0 0 0; color: rgba(255,255,255,0.3); font-size: 12px;">
                © ${new Date().getFullYear()} Zyora Labs, Tamil Nadu, India. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`
}

async function sendEmail(to: string, subject: string, htmlContent: string): Promise<boolean> {
  try {
    const response = await fetch(ZEPTO_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': ZEPTO_TOKEN,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        from: {
          address: SENDER_EMAIL,
          name: SENDER_NAME,
        },
        to: [
          {
            email_address: {
              address: to,
              name: to === ADMIN_EMAIL ? 'ZSE Admin' : undefined,
            },
          },
        ],
        subject: subject,
        htmlbody: htmlContent,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('ZeptoMail API error:', errorText)
      return false
    }

    return true
  } catch (error) {
    console.error('Error sending email:', error)
    return false
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: ContactFormData = await request.json()
    
    // Validate required fields
    if (!body.name || !body.email || !body.subject || !body.message) {
      return NextResponse.json(
        { success: false, error: 'All fields are required' },
        { status: 400 }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(body.email)) {
      return NextResponse.json(
        { success: false, error: 'Invalid email address' },
        { status: 400 }
      )
    }

    const subjectLabel = subjectLabels[body.subject] || body.subject

    // Send email to admin
    const adminEmailSent = await sendEmail(
      ADMIN_EMAIL,
      `[ZSE Contact] ${subjectLabel} from ${body.name}`,
      getAdminEmailTemplate(body)
    )

    // Send auto-reply to user
    const userEmailSent = await sendEmail(
      body.email,
      'Thank you for contacting ZSE - We received your message!',
      getUserAutoReplyTemplate(body)
    )

    if (!adminEmailSent) {
      return NextResponse.json(
        { success: false, error: 'Failed to send message. Please try again.' },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      message: 'Message sent successfully',
      autoReplySent: userEmailSent,
    })

  } catch (error) {
    console.error('Contact form error:', error)
    return NextResponse.json(
      { success: false, error: 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
